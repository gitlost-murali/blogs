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

In this post, we'll dive into 8-bit quantization, specifically exploring how the popular `bitsandbytes` library implements it, setting the stage for potentially optimizing it further with custom CUDA kernels using Triton.

# Level 1: 8-bit Quantization with `bitsandbytes`

The `bitsandbytes` library provides a user-friendly way to apply quantization, particularly its LLM.int8() scheme, which achieves significant memory reduction often with minimal impact on model accuracy for large models.

## How Does it Work? The Core Idea

The fundamental idea behind 8-bit quantization is to map the range of FP32 or FP16 values to a much smaller range representable by INT8 (typically -127 to 127). This mapping involves two key components:

1.  **INT8 Weights:** The original high-precision weights are converted into 8-bit integers.
2.  **Scaling Factor(s):** A floating-point number (often FP16 or FP32) is stored alongside the INT8 weights. This scaling factor is crucial for mapping the INT8 values back to their approximate original floating-point range during computation.

`bitsandbytes` employs **vector-wise quantization** for linear layers. This means that instead of using a single scaling factor for the entire weight matrix, it calculates a separate scaling factor for each row (or column, depending on the operation). This finer granularity helps preserve more information and maintain model accuracy compared to simpler per-tensor quantization.

## Row-wise quantization with example

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

So, after quantization, a linear layer's weights are stored in GRAM as:

*   An INT8 tensor (`weight.data`, also known as `weight.CB`).
*   A floating-point tensor containing the scaling factors for each vector (`weight.SCB`).

## A Simple Example (Conceptual Code)

Let's illustrate using a layer from a small model loaded via `transformers` and compare the `bitsandbytes` quantization with a conceptual "from-scratch" approach.

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb # We still need this to check the layer type

# --- Conceptual From-Scratch Quantization ---
# Note: This is simplified for illustration and doesn't include
# optimizations like outlier handling found in bnb.
def quantize_vectorwise(tensor, bits=8):
    if bits != 8: raise NotImplementedError("Only 8-bit supported")
    # Calculate scale (absolute max) per row (vector)
    # We use FP32 for scale calculation stability
    # Ensure tensor is on CPU for .float() if it's quantized
    tensor_float = tensor.cpu().float() if tensor.device.type != 'cpu' else tensor.float()
    scale = tensor_float.abs().max(dim=1, keepdim=True).values
    # Avoid division by zero
    scale = torch.where(scale == 0, torch.tensor(1.0, dtype=torch.float32), scale)
    scale = scale.to(tensor.device) # Move scale back to original device
    
    # Quantization formula: int_val = round(float_val * (max_int / scale))
    max_int = (2**(bits - 1)) - 1 # For INT8, this is 127
    
    # Ensure scale is broadcastable for the division
    # Perform calculation in FP32 for stability before converting to INT8
    quantized_tensor = (tensor.float() * max_int / scale.float()).round().clamp(-max_int-1, max_int).to(torch.int8)
    
    # Return INT8 weights and FP16 scale (common practice)
    return quantized_tensor, scale.half()

def dequantize_vectorwise(quantized_tensor, scale):
    # Dequantization formula: float_val = int_val * scale / max_int
    max_int = 127.0 # For INT8
    # Ensure scale is broadcastable and on correct device/dtype
    if scale.dim() == 1:
        scale = scale.unsqueeze(1)
    scale = scale.to(quantized_tensor.device).float()
    
    # Perform dequantization to FP16 (matching common practice)
    dequantized_tensor = (quantized_tensor.float() * scale / max_int).half()
    return dequantized_tensor

# --- Example Setup ---
model_name = "EleutherAI/gpt-neo-125M"
layer_to_inspect = "transformer.h.0.mlp.c_fc" # Example layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Loading model: {model_name}")

# --- Load Original FP16 Weights ---
# Load the model in its original precision (likely FP16 or FP32)
try:
    original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    # Extract the original weights for the layer we want to inspect
    layer_path = layer_to_inspect.split('.')
    original_layer = original_model
    for part in layer_path:
        original_layer = getattr(original_layer, part)
    original_weights = original_layer.weight.data.clone()
    print(f"Successfully loaded original FP16 weights for layer: {layer_to_inspect}")
except Exception as e:
    print(f"Could not load original model or extract layer: {e}")
    original_weights = None # Set to None if loading fails

# --- Apply From-Scratch Quantization (if original weights loaded) ---
if original_weights is not None:
    scratch_quantized_weights, scratch_scale = quantize_vectorwise(original_weights)
    print("--- From-Scratch Quantization --- ")
    print(f"  Weight dtype: {scratch_quantized_weights.dtype}")
    print(f"  Weight shape: {scratch_quantized_weights.shape}")
    print(f"  Scale dtype: {scratch_scale.dtype}")
    print(f"  Scale shape: {scratch_scale.shape}")
    print(f"  Example quantized weights (first 5 of first row):
{scratch_quantized_weights[0, :5]}")
    print(f"  Example scale (first 5 rows): {scratch_scale[:5].flatten()}")
else:
    print("Skipping from-scratch quantization as original weights couldn't be loaded.")

# --- Load bitsandbytes Quantized Model ---
print("\nLoading model with bitsandbytes 8-bit quantization...")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

try:
    bnb_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" # Handles placing layers on devices
    )
    
    # Access the quantized layer
    layer_path = layer_to_inspect.split('.')
    bnb_layer = bnb_model
    for part in layer_path:
        bnb_layer = getattr(bnb_layer, part)
        
    print(f"\n--- bitsandbytes Quantization (via transformers) ---")
    if isinstance(bnb_layer, bnb.nn.Linear8bitLt):
        # Access quantized weights (CB) and scaling factors (SCB)
        # Note: Weights might be on 'meta' device until first forward pass if using accelerate
        # We might need to move them explicitly for inspection if device_map was used.
        # Let's try accessing state if CB/SCB aren't directly on the weight
        if hasattr(bnb_layer.weight, 'CB') and bnb_layer.weight.CB is not None:
            bnb_quantized_weights = bnb_layer.weight.CB
            bnb_scale = bnb_layer.weight.SCB
        elif hasattr(bnb_layer, 'state') and hasattr(bnb_layer.state, 'CB') and bnb_layer.state.CB is not None:
             bnb_quantized_weights = bnb_layer.state.CB
             bnb_scale = bnb_layer.state.SCB
             print("  (Accessed CB/SCB via layer.state)")
        else:
            print("  Could not find CB/SCB directly or via state.")
            bnb_quantized_weights = None
            bnb_scale = None

        if bnb_quantized_weights is not None and bnb_scale is not None:
            # Ensure weights are on the expected device for printing
            bnb_quantized_weights = bnb_quantized_weights.to(device)
            bnb_scale = bnb_scale.to(device)
            
            print(f"  Layer Type: {type(bnb_layer)}")
            print(f"  Weight dtype: {bnb_quantized_weights.dtype}")
            print(f"  Weight shape: {bnb_quantized_weights.shape}")
            print(f"  Scale dtype: {bnb_scale.dtype}")
            print(f"  Scale shape: {bnb_scale.shape}")
            print(f"  Example quantized weights (first 5 of first row):
{bnb_quantized_weights[0, :5]}")
            print(f"  Example scale (first 5 rows): {bnb_scale[:5].flatten()}")
    else:
        print(f"  Layer {layer_to_inspect} is not a bnb.nn.Linear8bitLt instance. Type: {type(bnb_layer)}")

except Exception as e:
    print(f"Could not load quantized model or access layer: {e}")

# Note: Direct comparison of quantized values might show small differences due to specifics
# of implementation (e.g., exact rounding methods, handling of edge cases).
```
# The Forward Pass: Where Does De-quantization Happen?

This is the crucial part. We have INT8 weights and scaling factors sitting in GRAM. Neural network computations, especially matrix multiplications (GEMM), typically require FP16 or FP32 operands. So, how do we perform `output = activation @ weight.T + bias`?

A naive approach would be to de-quantize the *entire* weight matrix back to FP16/FP32 in GRAM before the multiplication:

```python
# Naive approach - NOT what bitsandbytes does efficiently
fp16_weight = weight.CB * weight.SCB / 127.0 # Dequantize
output = activation @ fp16_weight.T + bias 
```

This works, but it completely negates the memory savings, as we'd need temporary storage for the full-precision weights.

**The `bitsandbytes` Magic: De-quantization Inside the Kernel**

`bitsandbytes` avoids this massive overhead by performing the de-quantization *dynamically inside its custom CUDA kernel* during the matrix multiplication. Here's a breakdown of the LLM.int8() forward pass within `bnb.matmul`:

1.  **Input:** The kernel receives the input activation (usually FP16), the INT8 weights (`weight.CB`), and the FP16 scaling factors (`weight.SCB`).
2.  **Mixed-Precision Multiplication:**
    *   **Outlier Handling:** The kernel first identifies 'outliers' in the input activation tensor – values exceeding a predefined threshold (e.g., 6.0). These few outlier values are treated separately.
    *   **INT8 Path:** The non-outlier part of the activation is quantized to INT8 (using vector-wise quantization with dynamically computed scaling factors). This INT8 activation is then multiplied with the INT8 weights. This core multiplication happens using specialized INT8 compute units on the GPU (if available).
    *   **FP16 Path:** The activation outliers (kept in FP16) are multiplied by the corresponding slices of the weight matrix. To do this, the *required* INT8 weight slices are temporarily de-quantized to FP16 *within the kernel's registers or shared memory*, using their `SCB` scaling factors.
3.  **De-quantization & Combination:**
    *   The result of the INT8 multiplication is de-quantized back to FP16 using the weight and activation scaling factors. This again happens *within the kernel*, not in global GRAM.
    *   This de-quantized result is added to the result from the FP16 outlier path.
4.  **Output:** The final result (in FP16 or the specified compute data type) is returned.

**Key Takeaway:** The full-precision weights are *never* fully reconstructed in GPU global memory (GRAM). De-quantization occurs on-the-fly for the specific values needed during the computation, maximizing memory efficiency and leveraging fast INT8 compute capabilities.

# Pitching In: Can Triton Optimize This?

While `bitsandbytes` provides highly optimized CUDA kernels, Triton offers a Python-based framework for writing custom GPU kernels that can be easier to develop and iterate on than raw CUDA C++.

Could we implement a similar mixed-precision matrix multiplication using Triton? Conceptually, yes. A Triton kernel could:

1.  Accept FP16 activations, INT8 weights, and FP16 scaling factors as input.
2.  Load blocks of activation and weight data into shared memory or registers.
3.  Implement the outlier detection logic.
4.  Perform INT8 matrix multiplication for the non-outlier parts (potentially using Triton's built-in matrix multiplication operations, though INT8 support details would need checking).
5.  De-quantize necessary weight blocks within registers/shared memory for the outlier FP16 multiplication.
6.  Perform the on-the-fly de-quantization of the INT8 result.
7.  Combine the results and write the final FP16 output.

This could be an interesting exercise to:

*   Gain deeper insights into the quantization/de-quantization process.
*   Experiment with different block sizes or data loading strategies.
*   Potentially achieve comparable or even better performance for specific hardware or model architectures, although beating the heavily optimized `bitsandbytes` kernels would be challenging.

# Next Steps

We've explored the mechanics of 8-bit quantization using `bitsandbytes`, focusing on how it cleverly performs de-quantization within its CUDA kernels to maintain efficiency.

The next logical step, perhaps for Study Log 6, would be to:

1.  Implement a basic version of the 8-bit quantization and de-quantization logic from scratch in Python/PyTorch for a deeper understanding.
2.  Attempt to write a basic Triton kernel for a quantized matrix multiplication, comparing its structure and potential performance characteristics against the concepts discussed here.

Stay tuned! 