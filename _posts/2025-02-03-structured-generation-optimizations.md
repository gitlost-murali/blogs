---
title: "CUDA Study Log 4: Optimizing Constrained Decoding with CUDA"
excerpt: "Update traditional CUDA matrix multiplication kernel for constrained decoding"
date: 2025-02-27
categories:
  - Blog
tags:
  - LLM
  - CUDA
  - Optimization
  - Constrained Decoding
toc: true
header:
  teaser: "assets/images/struct-gen-triton-kernel/simple-logits-masking.svg"
---

# The Problem: Inefficient Computation in Constrained Decoding

Constrained decoding ensures language models generate outputs that follow specific patterns or schemas. This is crucial for tasks like API response generation or structured data creation where we need guaranteed syntactic correctness.

However, there's a significant computational inefficiency in standard constrained decoding:

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/simple-logits-masking.svg/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 1:</strong> Constrained decoding involves generating tokens that follow a specific schema or pattern.
    </p>
  </figcaption>
</figure>

During each generation step:

1. The model computes scores (logits) for every token in its vocabulary (typically 50k+ tokens)
2. We filter out tokens that would violate our schema/grammar
3. Only then do we sample from the allowed tokens


This means we're wasting computation on tokens we'll never use. For example, if we only need to generate "true" or "false", we still compute scores for all 50,000+ tokens in the vocabulary, only to use just two of them!


Let's explore three increasingly sophisticated approaches to optimize this process, starting from the simplest case to a fully dynamic CUDA-accelerated solution.

# The Three Levels of Optimization

Let's explore three increasingly sophisticated approaches to optimize this process:

1. **Compressing Finite State Machine**: Compress the FSM into a compact representation for faster state transitions
2. **Optimized Matrix Multiplication**: Only compute logits for allowed tokens
3. **Kernel Optimization**: Use Kernel to parallelize the logit computation



## 1. Compressing the Finite State Machine (FSM)

### Understanding Automata for Constrained Generation

Consider a simple binary classifier that outputs either "true", "false" or "NA".

The constrained decoding library [`outlines`](https://github.com/dottxt-ai/outlines) would convert this to an FSM graph:

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/sentence_automaton.png/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 2:</strong> The FSM for a binary classifier output.
    </p>
  </figcaption>
</figure>

In this automaton:

- Each state represents a step in the generation process
- The initial state (q0) has one transition: `"`
- The second state (q1) has three transitions: `true`, `false`, and `NA`
- The final state (q3) has one transition: `"`

__Key Optimization__: When states have only one possible transition (like q0 and q3), we can skip the generation step entirely and directly emit that token.
This reduces our generation steps from 3 to just 1, as we only need to actually generate at state q1.

Let's take another example from [SGLang paper](https://arxiv.org/pdf/2312.07104):

> The constant text sequence `{"summary": "` spans multiple tokens in the normal decoding process as shown in Fig. 3 (c), requiring multiple decoding stages, even though there is only one valid next token when decoding it. Therefore, the whole sequence can be decoded in a single step (i.e., forward pass). ([SGLang paper](https://arxiv.org/pdf/2312.07104))

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/sglang_fsm_compression.png/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 3:</strong> The decoding process of normal and compressed FSMs (the underscore_ means a space). <a href="https://arxiv.org/pdf/2312.07104">Source</a>
    </p>
  </figcaption>
</figure>

By the way, if you want to see how a pydantic schema is converted to an FSM, you can use the following code based on [`outlines`](https://github.com/dottxt-ai/outlines):

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/outlines-fsm-generation.png/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 4:</strong> FSM generation script using outlines.<a href="https://arxiv.org/pdf/2312.07104">Source</a>
    </p>
  </figcaption>
</figure>

## 2. Optimized Matrix Multiplication
Once we have an FSM, we can identify the allowed tokens for each state and only compute logits for those tokens.
Instead of the standard computation:

```python
logits = final_layer @ token_embeddings.T
```

We can optimize to:

```python
allowed_indices = fsm_index.get_allowed_tokens(fsm_state)
logits = final_layer @ token_embeddings[allowed_indices].T
```

__Performance Benefits__:

- __Memory reduction__: Only use embedding weights of allowed tokens. Reduced memory transfers between GRAM and processors/threads.
- __Computation reduction__: Matrix multiplication size dramatically reduced

## 3. Kernel based optimization

Instead of modifying the model architecture, we can implement dynamic filtering directly in the matrix multiplication kernel. This approach:

- Maintains the model's final layer unchanged
- Uses a CUDA kernel to filter logits during computation
- Reduces memory transfers between GPU memory and processors/threads


When implementing constrained decoding in CUDA, we need an efficient way to filter out tokens that aren't allowed by our finite state machine. Instead of computing logits for all tokens and then applying a mask (which wastes computation), we can filter at different levels of granularity during the matrix multiplication itself.


### 3.1 Block-level filtering

The first level of optimization can be done at the block level:

1. We maintain a binary mask of vocabulary size (128k) to indicate allowed tokens (1) and non-allowed tokens (0)
2. Before computing the matrix multiplication for a block, we check if any tokens in that block are allowed
3. If no tokens in the block are allowed, we skip the entire block's computation
4. This dramatically reduces unnecessary work for constrained generation

Let's start by taking standard matrix multiplication kernel from [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#final-result) and modify it to support the filtering of allowed tokens. In a way, this can be compared to Sparse Matrix Multiplication across columns of final layer weights `[768 x 128k]`.

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/block-level-filter.png/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 5:</strong> Block-level filtering.
    </p>
  </figcaption>
</figure>

```python
@triton.jit
def matmul_kernel(
        # Standard matrix multiply parameters
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        # New parameter: mask of allowed tokens
        allowed_cols_mask_ptr,
        # Other parameters...
):
    # Calculate block indices as usual
    # ...
    
    # Check if this block contains any allowed tokens
    allowed_cols_mask = tl.load(allowed_cols_mask_ptr + offs_bn, mask=offs_bn < N, other=0)
    block_has_valid_columns = tl.sum(allowed_cols_mask)
    
    # Skip block if no allowed tokens
    if block_has_valid_columns == 0:
        return
```

What's happening in the code:

1. We first determine which output matrix block is being computed by this CUDA block using `pid_m` and `pid_n`
2. We calculate the row indices of A (offs_am) and column indices of B (offs_bn) for the current block in the output matrix C
3. Block filtering step: From the token mask, we load a portion of the mask corresponding to the current block using `tl.load(allowed_cols_mask_ptr + offs_bn)`
4. We check if any tokens in this block are allowed by summing the mask values
5. If no tokens are allowed (block_has_valid_columns == 0), we skip the entire block computation by returning early
6. Otherwise, we proceed with the standard matrix multiplication for this block

This optimization allows us to skip entire blocks of computation when none of the tokens in that block are allowed by our FSM. For example, if each block has `BLOCK_SIZE_N = 32` and our allowed tokens are only [1, 5, 6, 20], only the first block will be active while all other blocks' computation will be skipped entirely. 

Ideally, out of the 32 threads in this block, only 4 threads should be active (corresponding to the allowed tokens) and the rest must be idle. However, for brevity, let's just go ahead and compute the entire block. Later, we will see how to handle this efficiently.

#### Filtering at the output level

Since we are computing the entire block of output, (32 columns instead of just 4), we need to filter the output at the end. This is done by using the mask `allowed_cols_mask` to filter the output.

<!-- Custom diff styling for code -->
<style>
  .diff-container {
    background-color: rgb(30, 29, 29);
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    font-family: "Source Code Pro", monospace;
    font-size: 0.4em; /* Uses relative sizing based on parent element */
    line-height: 1.5;
  }
  .diff-container pre {
    margin: 0;
    white-space: pre-wrap;
    color: white;
  }
  .inserted {
    background-color: #ddffdd;
    color: #333;
  }
</style>

<div class="diff-container">
  <pre><code>c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N) <span class="inserted">& (allowed_cols_mask[None, :] > 0)</span></code></pre>
</div>


### 3.2 Column-level/Thread-level filtering

Now that we have a kernel that can filter the output at the block level, we can extend it to filter the output at the column level. To address this, we can implement column-level filtering within each block:

1. We use the same allowed token mask but apply it at a finer granularity
2. When loading input data for each column, we check if that column corresponds to an allowed token
3. We only perform computations for the allowed columns. Others are set to 0, which is equivalent to skipping the computation.


<div class="diff-container">
  <pre><code>b = tl.load(b_ptrs, mask=((offs_k[:, None] < K - k * BLOCK_SIZE_K) <span class="inserted">& (allowed_cols_mask[None, :] != 0)</span>), other=0.0)</code></pre>
</div>

In the code above, we modify the mask condition when loading input data (b) to include `(allowed_cols_mask[None, :] != 0)`. This ensures we only load data for allowed columns, saving time on loading data for non-allowed columns.


### 3.3 The benchmarks

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/block-level-speed-ups.png/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 6:</strong> Block-level filtering speedups.
    </p>
  </figcaption>
</figure>

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/thread-level-speed-ups.png/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 7:</strong> Thread-level filtering speedups.
    </p>
  </figcaption>
</figure>


Based on the benchmarks, we can see that the block-level filtering provides the most significant speedup, followed by the thread-level filtering. The column-level filtering provides a modest speedup. The reason for this is that the block-level filtering is more efficient at reducing the number of computations, while the thread-level filtering is more efficient at reducing the number of memory transfers.

#### 3.3.1 Memory layout considerations

The efficiency of our approach depends heavily on how the allowed tokens are distributed. If allowed tokens are scattered randomly across the vocabulary, we might still need to process many blocks. However, in practice:
- For many constrained decoding scenarios, the number of allowed tokens is small compared to the vocabulary size.
- We can potentially reorder the vocabulary to cluster commonly allowed tokens together, improving block-level filtering efficiency. But this can get messy when we have multiple constraints.

### 3.4 Performance Comparison

The performance benefits of this approach are most significant when:
- The number of allowed tokens is small compared to the vocabulary size
- The allowed tokens are clustered together in the vocabulary
- The computation is memory-bound rather than compute-bound


## Conclusion

We've explored three levels of structured generation optimization:

1. Fixed structure provides the simplest optimization with significant speedup
2. Dynamic structure offers flexibility with moderate performance gains
3. CUDA implementation delivers maximum performance for complex cases

The choice between these approaches depends on your specific needs:
- For simple schemas with limited options, the fixed structure approach is sufficient
- For complex but predictable schemas, the dynamic approach provides a good balance
- For high-performance requirements or complex schemas, the CUDA implementation is optimal

Remember that these optimizations are complementary to the core benefits of structured generation: improved reliability and reduced hallucination. By making structured generation faster, we remove one of the main barriers to its adoption in production systems.

## Future Work

We're continuing to explore additional optimizations:
1. Batched CUDA kernels for multi-sequence generation
2. Pruning impossible states during initialization
3. Caching common state transitions
4. Hybrid approaches that combine different optimization strategies

Stay tuned for more updates as we push the boundaries of structured generation performance! 
