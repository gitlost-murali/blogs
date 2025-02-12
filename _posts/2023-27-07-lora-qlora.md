---
title: Understanding LoRA and QLoRA - The Powerhouses of Efficient Finetuning in Large Language Models
excerpt: Delving into the math behind LoRA and QLoRA 
tags: [Machine Learning, Language Models, QLoRA, LoRA, finetuning, huggingface, transformers]
date: 2023-07-27 05:28:10 +0530
categories: machine-learning data-science
toc: true
permalink: /:categories/:title
---

# Background

Large Language Models (LLMs) are currently a hot topic in the field of machine learning. Imagine you're an ML Engineer and your company has access to GPUs and open-source LLMs like LLAMA/Falcon. You're tasked with building tools for your customers, each with unique needs. You finetune your model for each customer, and everyone is satisfied.

But what happens when you have thousands of customers? Deploying thousands of GPU-hungry LLMs isn't feasible unless you have an extensive supply of GPUs. You need a strategy that allows the model to be finetuned for each customer without breaking the bank or overloading your storage. This is where QLoRA and LoRA come into play.

## Brief introduction to gradient descent

On a very abstract level, An LLM is essentially a function that takes some input, processes it and outputs something. We can represent it as f(x, W) = y, where x is the input sequence, y is the output sequence, and W is the set of weights of the model that are learned during training. W is black box that is doing the magic. 

These weights are large matrices. For instance, the weights of GPT-3 number 175 billion. What makes a perfect W? - I mean how do you find the perfect combination of parameters in W? You train the model on a dataset to __adjust the weights in W__ to minimize the difference between the output and expected output.

$$ W = W + \Delta W $$

where $\Delta W$ is the change in weights. We do this for a lot of iterations until we get a good W.

# LoRA (Low-Rank Adapters)

Instead of iteratively updating W in each step, what if we can store all those changes in $\Delta W$ and update W in one go? We can just store this $\Delta W$ for the finetuned task. When we want to perform inference for the intended task, we simply update W with $\Delta W$. Think of these $\Delta W$ as adaptable lenses that can be attached or detached to the base model as needed, allowing us to swiftly switch between tasks during inference.

Now, if W is 10000 x 10000, it means $\Delta W$ is also 10000 x 10000. We are taking space equivalent to the original model (W) to store $\Delta W$. This is a lot of memory. This is where LoRA comes into the picture. LoRA is a technique to reduce the memory footprint of $\Delta W$. It does this by using a low-rank approximation of $\Delta W$. This is done by decomposing $\Delta W$ into two matrices $W_{a}$ and $W_{b}$.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/big_matrix.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/big_matrix.png"></a>
    <figcaption><b>Figure 1:</b> <i> Big matrix <a href="https://www.youtube.com/watch?app=desktop&v=YVU5wAA6Txo">(Image Source)</a> </i></figcaption>
</figure>

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/svdmatrix.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/svdmatrix.png"></a>
    <figcaption><b>Figure 2:</b> <i> Big matrix decomposed into two matrices <a href="https://www.youtube.com/watch?app=desktop&v=YVU5wAA6Txo"> (Image Source)</a></i></figcaption>
</figure>

Let's break down everything step-by-step:

* If I have a matrix of size 2 x 2, it means 4 elements are stored in the memory. If the matrix is 100 x 100, it means alot of elements are stored in memory ($10000$). What if there's a better way to store the same information?? Here comes some inspiration from [Singular Value Decomposition (SVD)](https://www.geeksforgeeks.org/singular-value-decomposition-svd/),

    $$ \Delta W = W_{a} \times W_{b} $$

    where <br> 
    $\Delta W$ = $100 \times 100$ 

    $W_{a}$ = $100 \times 3$

    $W_{b}$ = $3 \times 100$. 

    Did you see what happened here? $W_{a} \times W_{b}$ gives you the original $100 \times 100$ matrix. This is a significant reduction in memory footprint. We are able to store the information of 10000 elements matrix with just two matrices 300 ($W_{a}$) & 300 ($W_{b}$), totalling just 600 elements ($W_{a} \times W_{b}$) in storage instead of 10000 elements.

* But how we did decide on 3? Why not 2, 1 or 68? To answer this question, we need to understand the rank of a matrix.
    * What is the Rank of a Matrix? - Rank of a matrix is the number of linearly independent rows/columns in a matrix. For example, if a matrix has 3 linearly independent rows, then the rank of the matrix is 3. If a matrix has 2 linearly independent columns, then the rank of the matrix is 2. 

    * What does linearly independent columns mean? Well, these represent factors of variation. In other words, these columns hold the most important factors that can help in uniquely representing the information. Let's say you have 10 x 10 matrix with 4 linearly independent columns, then there are 4 factors of variation in the matrix. If the rank is 4, it means we have 6 redundant columns.

    * Think of it this way, do we really think we need 175 billion parameters for a small task for summarization? No, right? We can do with a lot less. This is where the rank comes into the picture. We can reduce the rank to 1000 or 100 or 10. This will reduce the memory footprint of $\Delta W$.


    * Of course, there is a catch when we consider low rank. We are approximating the gradient $\Delta W$ here. Hence, the name Low-Rank approximation. __Select your rank based on the downstream task__. If you think a task requires __less IQ__, __reduce the rank__. Otherwise, increase the rank to hold more information.

The essence of LoRA is that we can freeze the W and just update     $W_{a}$ and $W_{b}$. $W_{a} \times W_{b}$ will give you the updated $\Delta W$. After finetuning, we can update the W with the new $\Delta W$.


$$ W = W + \Delta W $$

becomes

$$ W = W +  W_{a} \times W_{b} $$

We are bypassing the step of storing large $\Delta W$ into the memory. This is the benefit of using LoRA. Just store the matrices  $ W_{a} \& W_{b} $ into your disk, which would be maybe 1% of the original model weights (incase of $W$ being 175B params and $\Delta W$ being 1B params). So, if you have 1000 customers and need 1000 tasks, we can just store 1000 $W_{a}$ and 1000 $W_{b}$ matrices, which are way smaller than the original model weights. For inference, load the original model weights once and then load the $W_{a}$ and $W_{b}$ matrices for each task. This is a huge reduction in memory footprint.

## Let's bring it to code

Any guesses?

```python
def regular_forward_matmul(x, W):
    h = x @ W
return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # regular matrix multiplication
    h += x @ (W_A @ W_B) # updated equation
return h
```

We just added $x @ (W_A @ W_B)$ to the existing equation. Since we are freezing W, the only thing that needs gradient updates are $(W_A \& W_B)$. The final weights $ W_{a} \times W_{b} $ are the delta weights $\Delta W$ we need for our finetuned task.

## LoRA in Transformers

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, # rank
    lora_alpha=16, # lora scaling factor
    target_modules=["query", "value"], # modules to apply LoRA
    lora_dropout=0.1, # dropout
    bias="none",
    modules_to_save=["classifier"], # additional modules to save
)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
lora_model = get_peft_model(model, config)
```

In the above code, we are assigning the lora rank `r` to 16. `lora_alpha` is the scaling factor that determines how much importance you want to give to the new updated $\Delta W$ i.e $ W_{a} \times W_{b}$ when adding it to the original pretrained weights $W$. The `target_modules` are the modules where we want to apply LoRA. In this case, we are applying LoRA to the query and value modules. The bias is the bias term in the linear layer. We can set it to none or true. If we set it to none, we are not using bias. If we set it to true, we are using bias. The modules_to_save are the additional modules we want to save. In this case, we are saving the classifier module. 


# QLoRA (Quantized LoRA)

While LoRA helps in reducing the storage requirements, you would still need a large GPU to load the model into the memory for LoRa training. This is where QLoRA, or Quantized LoRA, comes into the picture. QLoRA is a combination of LoRA and Quantization.

Currently, we store the weight parameters in FP32. What does it mean? Each element in the matrix is stored in 32 bits. What if we can store the same information in 8 bits? 4 bits? Before I throw some math at you, let me give you a brief overview of QLoRa. 

__QLoRA:__ Here, you first quantize the LLM and then perform LoRa training. That's it.

Here are some more details to the last statement:

1. Quantize the LLM to 4 bits (NF4). This means that each element in the matrix is stored in 4 bits. This is a huge reduction in memory footprint.
2. __Next__, we perform LoRa training in 32 bit precision (FP32).
3. At first glance, it may seem counterintuitive to quantize the model to 4 bits and then perform LoRa training in 32 bits. However, this is a necessary step. To train LoRa adapters in FP32, the model __weights must be returned to FP32__ as well. This process involves reversing the quantization (de-quantization), which is done in a step-by-step manner.
4. One might assume that de-quantization to FP32 would cause an explosion in GPU VRAM. However, this is not the case. Consider the model as a large sheet of paper.
    <figure>
        <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/4bitqlora.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/4bitqlora.png"></a>
        <figcaption><b>Figure 3:</b> <i>Quantized model before computation. 4bit elements are model weights and 32 bits are LoRa weights (Wa and Wb)</i></figcaption>
    </figure>
5. Now, think of the forward pass like a torchlight applied on a big sheet of paper. Wherever the torch is applied, the 4 bit elements are converted to 32 bit elements. We are __converting__ the 4 bit elements to 32 bit elements __only when we need them__. And once the computation is done, they are back to 4 bits.

    <figure>
        <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/32bitqlora.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/32bitqlora.png"></a>
        <figcaption><b>Figure 4:</b> <i>Computation step: 4bit model weights are converted to 32 bits during forward pass and backpropagation steps</i></figcaption>
    </figure>

6. In this approach, only the LoRa adapters are stored in FP32 format, while the rest remain in 4-bit format. This strategy results in a significant reduction in memory footprint.

The section below explains the math behind NF4 quantization. You can skip to the code section if you're allergic to math.

## NF4 Quantization

If you have 32 bits to store information, you can store $ 2^{32} $ values. However, if you can store the same information in 8 bits (range of -127 to 127), you can drastically reduce the memory requirements. What if it's only 4 bits?? 


The paper says the following:

1. 4-bit integers represent 16 levels which are evenly spaced in the [−1, 1] range. The levels would be
-1.0, -0.8667, -0.7333, -0.6, -0.4667, -0.3333, -0.2, -0.0667, 0.0667, 0.2, 0.3333, 0.4667, 0.6, 0.7333, 0.8667, 1.0
2. Let's say a weight in the big FP32 model is 0.23456.
3. The closest value in the 16 levels is 0.2.
4. So, we quantize the weight to 0.2.
5. In our 4-bit representation, we store the value 10 (0.2 is the 10th value in the 16 levels).
6. If we want to use this 4-bit weight in computation, we dequantize it back to FP32 using the index stored. (10th index = 0.2)
7. The dequantization error is 0.23456 - 0.2 = 0.03456 (~1/4th of the quantization step size - 0.1333).

This is a simplified explanation of the process. In practice, the NF4 quantization technique involves other steps, such as splitting 16 levels with quartiles, normalizing input tensor, etc. to ensure that the quantized values accurately represent the original data.

Let's answer why we want to have FP32 precision for LoRa adapters. The quantization and de-quantization results in loss of information in the model weights. Maintaining the LoRa adapters in FP32 precision ensures that the loss of information is subdued and higher precision allows the low-rank adapters to capture subtle nuances in the downstream task they are trained for.

## Code with Transformers

```python
from transformers import BitsAndBytesConfig


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
   #FP16 for faster tuning. You can also choose FP32 for higher precision
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

# Conclusion

This is the end of the blog. I hope you enjoyed reading it. If you have any questions, please feel free to reach out on [Linkedin](https://www.linkedin.com/in/murali-manohar/), [Twitter](https://twitter.com/gitlostmurali) or [Mail](mailto:kmanoharmurali@gmail.com).
