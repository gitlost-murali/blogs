---
title: Demystifying LoRA and QLoRA - The Powerhouses of Efficient Finetuning in Large Language Models
excerpt: Delving into the math behind LoRA and QLoRA 
tags: [Machine Learning, Language Models, QLoRA, LoRA, finetuning, huggingface, transformers]
date: 2023-07-27 05:28:10 +0530
categories: machine-learning data-science
toc: true
permalink: /:categories/:title
---

## Introduction

Large Language Models (LLMs) are currently a hot topic in the field of machine learning. Imagine you're an ML Engineer and your company has access to GPUs and open-source LLMs like LLAMA/Falcon. You're tasked with building tools for your customers, each with unique needs. You finetune your model for each customer, and everyone is satisfied.

But what happens when you have thousands of customers? Deploying thousands of GPU-hungry LLMs isn't feasible unless you have an extensive supply of GPUs. You need a strategy that allows the model to be finetuned for each customer without breaking the bank or overloading your storage. This is where QLoRA and LoRA come into play.

## QLoRA and LoRA

Let's get going. On a very abstract level, An LLM is essentially a function that takes some input, processes it and outputs something. For brevity, let's call it as $f(x, W) = y$, where x is the input sequence, y is the output sequence and W is black box that is doing the magic. Essentially, W is the set of weights of the model that are learned during training.

These weights are matrices, big big bigggg matrices. For example, the weights of GPT-3 are 175 billion in number - meaning the total elements in all the matrices are 175 billion.

What makes a perfect W? - I mean how do you find the perfect combination of parameters in W? You train the model on a dataset to __adjust the weights in W__ to minimize the difference between the output and expected output. This is called training.

$$ W = W + \Delta W $$

where $\Delta W$ is the change in weights. We do this for a lot of iterations until we get a good W.

## LoRa

Now, if W is 10000 x 10000, it means $\Delta W$ is also 10000 x 10000. This is a lot of memory. This is where LoRA comes into the picture. LoRA is a technique to reduce the memory footprint of $\Delta W$. It does this by using a low-rank approximation of $\Delta W$. This is done by decomposing $\Delta W$ into two matrices $W_{a}$ and $W_{b}$.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/big_matrix.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/big_matrix.png"></a>
    <figcaption><b>Figure 3:</b> <i> Big matrix <a href="https://www.youtube.com/watch?app=desktop&v=YVU5wAA6Txo">(Image Source)</a> </i></figcaption>
</figure>

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/svdmatrix.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/svdmatrix.png"></a>
    <figcaption><b>Figure 4:</b> <i> Big matrix decomposed into two matrices <a href="https://www.youtube.com/watch?app=desktop&v=YVU5wAA6Txo"> (Image Source)</a></i></figcaption>
</figure>

Let's break down everything step-by-step:

* If I have a matrix of size 2 x 2, it means 4 elements are stored in the memory. If the matrix is 100 x 100, it means alot of elements are stored in memory ($10000$). What if we there's a better way to store the same information?? Here comes SVD,

$$ \Delta W = W_{a} \times W_{b} $$

where <br> 
$\Delta W$ = $100 \times 100$ 

$W_{a}$ = $100 \times 3$

$W_{b}$ = $3 \times 100$. 

Did you see what happened here? $W_{a} \times W_{b}$ gives you the original $100 \times 100$ matrix. Great!! This is a huge reduction in memory footprint. We are able to store the information of 10000 elements matrix with just two matrices 300 ($W_{a}$) & 300 ($W_{b}$), totalling just 600 elements ($W_{a} \times W_{b}$) in storage instead of 10000 elements.

* But how we did decide on 3? Why not 2, 1 or 68? Well, this is where the rank of a matrix comes into the picture.

* What is the Rank of a Matrix? - Rank of a matrix is the number of linearly independent rows/columns in a matrix. For example, if a matrix has 3 linearly independent rows, then the rank of the matrix is 3. If a matrix has 2 linearly independent columns, then the rank of the matrix is 2. 

* What does linearly independent columns mean? Well, these represent factors of variation. In other words, these columns hold the most important factors that can help in uniquely representing the information. Let's say you have 10 x 10 matrix with 4 linearly independent columns, then there are 4 factors of variation in the matrix. If the rank is 4, it means we have 6 redundant columns.

* Think of it this way, do you really think we need 175 billion parameters? Let's say it has AGI shit level knowledge in it. But if you are finetuning it for a downstream task/domain, only a few parameters are needed for downstream task.

This is the essence of LoRA. Ofcourse, there is a catch when we consider low rank. We are approximating the gradient $\Delta W$ here. Hence, the name Low-Rank approximation. It is fine. Select your rank based on the downstream task. If you think that task requires less IQ, reduce the rank. Otherwise, increase the rank to hold more information.

Now that we know this information, if we want to finetune the LLM on a downstream task, we can freeze the W and just update $W_{a}$ and $W_{b}$. $W_{a} \times W_{b}$ will give you the updated $\Delta W$. After finetuning, we can update the W with the new $\Delta W$.

$$ W = W + \Delta W $$

becomes

$$ W = W +  W_{a} \times W_{b} $$

How does this benefit us? well, we are bypassing the step of storing large $\Delta W$ (10000) into the memory. This is the essence of LoRA. Just store the matrices  $ W_{a} \& W_{b} $ into your disk, which would be maybe 1% of the original model weights. So, if you have 1000 customers and need 1000 tasks, we can just store 1000 $W_{a}$ and 1000 $W_{b}$ matrices, which are way smaller than the original model weights. For inference, load the original model weights once and then load the $W_{a}$ and $W_{b}$ matrices for each task. This is a huge reduction in memory footprint.

### Let's bring it to code

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

Did you see what we did here? We added $x @ (W_A @ W_B)$ to the existing equation. Since we are freezing W, the only thing that needs gradient updates are $(W_A \& W_B)$. The final weights $ W_{a} \times W_{b} $ are the delta weights $\Delta W$ we need for our finetuned task.

### LoRA in Transformers

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

In the above, we are assigning the lora rank `r` to 16. `lora_alpha` is the scaling factor that determines how much importance you want to give to the new updated $\Delta W$ i.e $ W_{a} \times W_{b}$ when adding it to the original pretrained weights $W$. The `target_modules` are the modules where we want to apply LoRA. In this case, we are applying LoRA to the query and value modules. The bias is the bias term in the linear layer. We can set it to none or true. If we set it to none, we are not using bias. If we set it to true, we are using bias. The modules_to_save are the additional modules we want to save. In this case, we are saving the classifier module. 


## QLoRA

Although you can store the finetuned weights of a 33B model in the disk, you would still need a big GPU to load the 33B model into the memory to perform LoRa training. You would have to be rich to save money. Bwahaha

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/jinyang_1.gif"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/jinyang_1.gif"></a>
    <figcaption><b>Figure 5:</b> <i>Jian Yang says hello</i></figcaption>
</figure>

Worry not. QLoRa to the rescue. Currently, we store the weight parameters in FP32. What does it mean? Each element in the matrix is stored in 32 bits. What if we can store the same information in 8 bits? 4 bits? This is where QLoRa comes into the picture. QLoRa is Quantized LoRa. It is a combination of LoRa and Quantization. Before I throw some math at you, let me give you a brief overview of QLoRa. 

__QLoRA:__ Well, first you quantize the LLM and then perform LoRa training. That's it.

Here are some more details to the last statement:

1. Quantize the LLM to 4 bits (NF4). This means that each element in the matrix is stored in 4 bits. This is a huge reduction in memory footprint.
2. __Next__, we perform LoRa training in 32 bit precision (FP32).
3. Isn't that weird? We quantized the model to 4 bits and then we are performing LoRa training in 32 bits. How does that work? For us to train LoRa adapters in FP32, we need the __model weights back in FP32 too__. We will have to undo the quantization. __Step by Step__.
4. But if you undo quantization, your GPU VRAM will explode? Not really. Think of your model as a big sheet of paper like below.

    <figure>
        <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/4bitqlora.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/4bitqlora.png"></a>
        <figcaption><b>Figure 4:</b> <i>Quantized model before computation. 4bit elements are model weights and 32 bits are LoRa weights (Wa and Wb)</i></figcaption>
    </figure>

5. Now, think of the forward pass like a torchlight applied on a big sheet of paper. Wherever the torch is applied, the 4 bit elements are converted to 32 bit elements. We are __converting__ the 4 bit elements to 32 bit elements __only when we need them__. And once the computation is done, they are back to 4 bits.

    <figure>
        <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/32bitqlora.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/32bitqlora.png"></a>
        <figcaption><b>Figure 5:</b> <i>Computation step: 4bit model weights are converted to 32 bits during forward pass and backpropagation steps</i></figcaption>
    </figure>

6. In this approach, you only store the LoRA adapters in FP32 format and the rest in 4 bit format. This is a huge reduction in memory footprint.

The section belows explains the math behind NF4 quantization. You can skip to the code section if you're allergic to math.

### NF4 Quantization

If you have 32 bits to store information, you can store $ 2^{32} $ values. However, if you can store the same information in 8 bits (range of -127 to 127), you can drastically reduce the memory requirements. What if it's only 4 bits?? 

__NF4__

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
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

## Badum Tss

This is the end of the blog. I hope you enjoyed reading it. If you have any questions, please feel free to reach out by clicking on the social media icons on the left. :)