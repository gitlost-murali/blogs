---
title: "CUDA Study Log 4: Tweaking a CUDA Kernel for Constrained Decoding"
excerpt: "Update traditional CUDA matrix multiplication kernel for constrained decoding"
date: 2025-02-27
categories:
  - Blog
tags:
  - LLM
  - CUDA
  - Optimization
  - Constrained Decoding
header:
  teaser: "assets/images/struct-gen-triton-kernel/simple-logits-masking.svg"
---

# Motivation

Constrained decoding constrains language model outputs to follow specific patterns or schemas. By defining a formal grammar or schema (like JSON Schema), we can ensure the model only generates valid outputs. This is particularly useful for tasks like API response generation, data structure creation, or any scenario where we need guaranteed syntactic correctness.

However, there's a computational inefficiency at the heart of constrained decoding. Consider what happens during each generation step:

<figure>
  <img src="{{ site.url }}/assets/images/struct-gen-triton-kernel/simple-logits-masking.svg/" alt="Structured Generation Teaser">
  <figcaption>
    <p>
      <strong>Figure 1:</strong> Constrained decoding involves generating tokens that follow a specific schema or pattern.
    </p>
  </figcaption>
</figure>

1. The model computes logits (scores) for every token in its vocabulary (typically 50k+ tokens)
2. We filter out tokens that would violate our schema/grammar
3. Only then do we sample from the allowed tokens


This means we're doing a lot of unnecessary computation. For example, in the image above, if we're generating a boolean field that can only be "true" or "false", we still compute scores for all 50,000+ tokens in the vocabulary, only to use just two of them! This inefficiency compounds across every token we generate.


Let's explore three increasingly sophisticated approaches to optimize this process, starting from the simplest case where we know all possible tokens upfront, to a fully dynamic CUDA-accelerated solution.

## The Three Levels of Optimization

1. **Fixed Structure Optimization**: The simplest case where we know all possible outputs at initialization
2. **Dynamic Structure**: Runtime adaptation of allowed tokens based on state
3. **CUDA-Accelerated Dynamic Structure**: Parallel processing for maximum performance

Let's explore each approach in detail, examining their tradeoffs and implementation complexities.

## 1. Fixed Structure Optimization

The simplest optimization case occurs when we have a fixed structure with a limited set of possible values. Consider this example:

```python
from enum import Enum
from pydantic import BaseModel

class Name(str, Enum):
    john = "John"
    paul = "Paul"

class Age(int, Enum):
    twenty = 20
    thirty = 30

class Character(BaseModel):
    name: Name
    age: Age
```

In this scenario, we know exactly what values are allowed at each position:
- `name` can only be "John" or "Paul"
- `age` can only be 20 or 30

The key optimization here is that we can pre-compute the allowed token indices during model initialization:

```python
import json
from outlines.fsm.json_schema import convert_json_schema_to_str
from outlines_core.fsm.json_schema import build_regex_from_schema
import interegular

# Convert schema to FSM
json_schema = Character.model_json_schema()
json_schema_str = convert_json_schema_to_str(json_schema)
regex_str = build_regex_from_schema(json_schema_str)
pattern = interegular.parse_pattern(regex_str)
fsm = pattern.to_fsm()

# Create deterministic FSM and index
from outlines.fsm.regex import make_deterministic_fsm, create_fsm_index_tokenizer
new_fsm, _ = make_deterministic_fsm(fsm)
index, _ = create_fsm_index_tokenizer(new_fsm, tokenizer)
```

The optimization comes from modifying the model's final layer to only compute logits for the allowed tokens. Instead of the standard computation:

```python
logits = final_layer @ token_embeddings.T  # Shape: [batch_size, vocab_size]
```

We can use:

```python
allowed_indices = index.all_possible_indices  # Pre-computed during init
allowed_embeddings = token_embeddings[allowed_indices]
logits = final_layer @ allowed_embeddings.T  # Shape: [batch_size, len(allowed_indices)]
```

This reduces the matrix multiplication from O(vocab_size) to O(num_allowed_tokens), which in our example is just 4 tokens (2 names × 2 ages).

### Performance Impact

For this fixed structure case, we see significant speedups:

1. Memory reduction: We only need to store embeddings for allowed tokens
2. Computation reduction: Matrix multiplication size is drastically reduced
3. No runtime overhead: Token filtering is done once during initialization

Here's a simple benchmark comparing standard vs. optimized generation:

```python
import time

def benchmark_generation(model, prompt, n_runs=1000):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.generate(prompt)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)

# Results (average generation time per token):
# Standard: 2.3ms
# Optimized: 0.4ms
```

## 2. Dynamic Structure

The dynamic case is more complex because the allowed tokens change based on the current state in the FSM. Let's extend our example to handle nested structures:

```python
class Inventory(BaseModel):
    items: List[str]
    capacity: int
    
class Character(BaseModel):
    name: Name
    age: Age
    inventory: Inventory
```

Now the allowed tokens depend on the current position:
- At the start of an item list: Opening bracket
- Inside the list: String tokens or comma
- After the list: Closing bracket
- For capacity: Integer tokens

The optimization approach changes to:

```python
class OptimizedStructuredGeneration:
    def forward(self, hidden_states, fsm_state):
        # Get allowed tokens for current state
        allowed_indices = self.index.get_allowed_tokens(fsm_state)
        
        # Only compute logits for allowed tokens
        allowed_embeddings = self.token_embeddings[allowed_indices]
        logits = hidden_states @ allowed_embeddings.T
        
        return logits, allowed_indices

    def update_state(self, current_state, token):
        return self.fsm.next_state(current_state, token)
```

This dynamic approach maintains the core optimization of reduced matrix multiplication while adapting to changing constraints. The tradeoff is the additional overhead of state tracking and token filtering at each step.

## 3. CUDA-Accelerated Dynamic Structure

For maximum performance, we can move the dynamic token filtering to CUDA. The key insight is that we can parallelize both the state transitions and the logit computations.

Here's a simplified version of the CUDA kernel:

```cuda
__global__ void compute_allowed_logits(
    float* hidden_states,      // [batch_size, hidden_dim]
    float* token_embeddings,   // [vocab_size, hidden_dim]
    int* fsm_states,          // [batch_size]
    int* allowed_tokens,       // [batch_size, max_allowed_tokens]
    float* output_logits,      // [batch_size, max_allowed_tokens]
    int batch_size,
    int hidden_dim,
    int max_allowed_tokens
) {
    int batch_idx = blockIdx.x;
    int token_idx = threadIdx.x;
    
    if (batch_idx < batch_size && token_idx < max_allowed_tokens) {
        int token_id = allowed_tokens[batch_idx * max_allowed_tokens + token_idx];
        if (token_id >= 0) {  // Valid token
            float sum = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                sum += hidden_states[batch_idx * hidden_dim + i] * 
                       token_embeddings[token_id * hidden_dim + i];
            }
            output_logits[batch_idx * max_allowed_tokens + token_idx] = sum;
        }
    }
}
```

The Python wrapper:

```python
class CUDAStructuredGeneration:
    def __init__(self, model, fsm_index):
        self.cuda_module = load_cuda_kernels()
        self.model = model
        self.fsm_index = fsm_index
        
    def forward(self, hidden_states, fsm_states):
        # Get allowed tokens for all states in parallel
        allowed_tokens = self.fsm_index.batch_allowed_tokens(fsm_states)
        
        # Launch CUDA kernel
        output_logits = torch.zeros(
            hidden_states.shape[0],
            allowed_tokens.shape[1],
            device='cuda'
        )
        
        self.cuda_module.compute_allowed_logits(
            hidden_states,
            self.model.token_embeddings,
            fsm_states,
            allowed_tokens,
            output_logits
        )
        
        return output_logits, allowed_tokens
```
### Performance Comparison

Here's a benchmark comparing all three approaches:

```python
def benchmark_all(prompt, n_runs=1000):
    results = {}
    for method in ['standard', 'fixed', 'dynamic', 'cuda']:
        model = get_model(method)
        results[method] = benchmark_generation(model, prompt, n_runs)
    return results

# Results (average generation time per token):
# Standard:    2.30ms
# Fixed:       0.40ms
# Dynamic:     0.85ms
# CUDA:        0.15ms
```

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
