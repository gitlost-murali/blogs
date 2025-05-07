---
title: "CUDA Study Log 5: Demystifying 8-bit Quantization with bitsandbytes"
excerpt: "Understanding the mechanics of 8-bit quantization and the role of CUDA kernels in bitsandbytes for efficient LLM inference."
date: 2025-03-09
categories:
  - Blog
tags:
  - LLM
  - CUDA
  - Optimization
  - Quantization
  - bitsandbytes
  - Triton
toc: true
header:
  teaser: "assets/images/placeholder_quantization.png" # TODO: Replace with a relevant image
---

# The Need for Speed (and Memory): Why Quantization Matters

In this post, we'll dive into 8-bit quantization, specifically exploring how the popular `bitsandbytes` library implements it, setting the stage for our studylog's mission: Understanding/Writing CUDA Kernels.

# Level 1: 8-bit Quantization with `bitsandbytes`

The `bitsandbytes` library provides a user-friendly way to apply quantization, particularly its LLM.int8() scheme, which achieves significant memory reduction often with minimal impact on model accuracy for large models.

## How Does it Work? The Core Idea

The fundamental idea behind 8-bit quantization is to map the range of FP32 or FP16 values to a much smaller range representable by INT8 (typically -127 to 127). This mapping involves two key components:

1.  **INT8 Weights:** The original high-precision weights are converted into 8-bit integers.
2.  **Scaling Factor(s):** A floating-point number (often FP16 or FP32) is stored alongside the INT8 weights. This scaling factor is crucial for mapping the INT8 values back to their approximate original floating-point range during computation.

`bitsandbytes` employs **vector-wise quantization** for linear layers. This means that instead of using a single scaling factor for the entire weight matrix, it calculates a separate scaling factor for each row (or column, depending on the operation). This finer granularity helps preserve more information and maintain model accuracy compared to simpler per-tensor quantization.

## Row-wise Quantization: A Worked Example

Let's quantize a simple 2×3 tensor row-wise to 8 bits:

```
X = [[2.5,  -6.1,  1.0],
     [0.3,   0.2, -0.1]]
```

### Step 1: Calculate scales (max absolute value per row)
- Row 1: 
$s_1 = \max(|2.5|, |-6.1|, |1.0|) = 6.1$
- Row 2:
$s_2 = \max(|0.3|, |0.2|, |-0.1|) = 0.3$

### Step 2: Quantize each element
For 8-bit, $N = 127$

Row 1:
- $Q_{1,1} = \text{round}(\frac{2.5 \times 127}{6.1}) = \text{round}(52.05) = 52$
- $Q_{1,2} = \text{round}(\frac{-6.1 \times 127}{6.1}) = \text{round}(-127) = -127$
- $Q_{1,3} = \text{round}(\frac{1.0 \times 127}{6.1}) = \text{round}(20.82) = 21$

Row 2:
- $Q_{2,1} = \text{round}(\frac{0.3 \times 127}{0.3}) = \text{round}(127) = 127$
- $Q_{2,2} = \text{round}(\frac{0.2 \times 127}{0.3}) = \text{round}(84.67) = 85$
- $Q_{2,3} = \text{round}(\frac{-0.1 \times 127}{0.3}) = \text{round}(-42.33) = -42$

Resulting in quantized tensor:
```
Q = [[52, -127, 21],
     [127, 85, -42]]
```

With scales:
```
s = [[6.1],
     [0.3]]
```


# The Forward Pass: Where Does De-quantization Happen?

After quantization, a linear layer's weights are stored in GPU RAM (GRAM) as:

* An `INT8` tensor containing the quantized weights (often called `weight.CB` in bitsandbytes)
* A floating-point tensor (`FP16` or `FP32`) containing the scaling factors for each row (`weight.SCB`)


This is the crucial part. We have INT8 weights and scaling factors sitting in GRAM. Neural network computations, especially matrix multiplications (GEMM), typically require FP16 or FP32 operands. So, how do we perform `output = input_x @ weight.T + bias`?

## The Naive Approach: Full Dequantization

One approach would be to dequantize the entire weight matrix back to `FP16`/`FP32` before performing the matrix multiplication:

```python
# Naive approach - NOT what bitsandbytes does efficiently
fp16_weight = quantized_w * scale / 127.0 # Dequantize
output = input_x @ fp16_weight.T + bias 
```

This works, but it completely defeats the purpose of quantization for inference! We'd need to allocate temporary storage for the full-precision weights, negating any memory savings we hoped to achieve.

## **The `bitsandbytes` approach: De-quantization Inside the Kernel**

`bitsandbytes` avoids this massive overhead by performing the de-quantization *dynamically inside its custom CUDA kernel* during the matrix multiplication. Here's a breakdown of the LLM.int8() forward pass within `bnb.matmul`:


1. **Input Preparation:** The CUDA kernel receives:
    a. The input activation tensor (typically in FP16)
    b. The quantized INT8 weights (`weight.CB`)
    c. The FP16 scaling factors for each row (`weight.SCB`)

2. **On-the-fly Computation:** The kernel then:

    a. Loads blocks of the INT8 weights into fast shared memory or registers
    b. Dequantizes the INT8 weights into FP16 and performs matmul.
        > Note that latest GPUs also support performing `INT8` operations natively but we would have to bring the input activations also to `INT8`, which (TODO: reason why not)
    d. Accumulates the final results in FP16 or FP32

3. **Output:** Returns the computed activations in the desired precision


## Kernel

For brevity, we are considering weight-only quantization with compute data-type as `float16`. We are omitting the case of native `INT8` support.

I believe we can take the standard matmul kernel and tweak it a little bit to achieve what we want. Here's what I mean

```

```