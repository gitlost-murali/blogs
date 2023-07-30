---
title: QLoRA and LoRA - Revolutionizing Finetuning of Large Language Models
excerpt: Delving into the innovative techniques of QLoRA and LoRA for efficient finetuning of large language models (LLMs)
tags: [Machine Learning, Language Models, QLoRA, LoRA, Finetuning]
date: 2023-07-27 05:28:10 +0530
categories: machine-learning data-science
toc: true
permalink: /:categories/:title
---

## Introduction

LLMs are the buzz now. You are a ML Engineer in a company. Let's say your silicon valley CEO comes to you saying, "Hey, we got the GPUs and there are open-source LLMs like LLAMA/Falcon. Let's build tools for our customers". Each vendor/customer has a different need. You finetune your model for each customer. You are happy. CEO is happy. Customers are happy.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/gbelson.jpg"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/gbelson.jpg"></a>
    <figcaption><b>Figure 1:</b> <i>Let's get those hotcakes called LLMs</i></figcaption>
</figure>

End credits already? But, what if you have 1000s of customers? You can't deploy those 1000 clones of those GPU hungry LLMs. Unless you have a Gilfoyle in your team with loads of GPUs, you can't afford to do that.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/gilfoyle_servers.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/gilfoyle_servers.png"></a>
    <figcaption><b>Figure 2:</b> <i>Get Gilfoyle asap</i></figcaption>
</figure>

You need a generic model that can be finetuned for each customer. This is where QLoRA and LoRA come into the picture.

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

Let's break down the NF4 quantization process with a simple vector and some math.

Consider a vector v:
```shell
v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

The first step in the NF4 quantization process is to determine the minimum and maximum values of the vector. Let's denote these as v_min and v_max respectively.

```shell
v_min = min(v) = 0.1
v_max = max(v) = 1.0
```

Next, we calculate the range of the vector, which is the difference between the maximum and minimum values.

```shell
range = v_max - v_min = 1.0 - 0.1 = 0.9
```

Since we are quantizing to 4 bits, we have 2^4 = 16 different levels that each value can be mapped to. We divide the range of the vector into 16 equal intervals to determine these levels.

```shell
interval = range / 16 = 0.9 / 16 = 0.05625
```

Here are the 16 levels represented as a vector with the interval as above:

```shell
levels = [0.1, 0.15625, 0.2125, 0.26875, 0.325, 0.38125, 0.4375, 0.49375, 0.55, 0.60625, 0.6625, 0.71875, 0.775, 0.83125, 0.8875, 0.94375]
```


Now, we map each value in the vector to the nearest quantization level. This involves subtracting the minimum value from each element, dividing by the interval size, and rounding to the nearest integer.

```shell
quantized_v = round((v - v_min) / interval) - 8
```

This will give us a new vector where each value is an integer between -8 and 7, represented by 4 bits. These are essentially indices that we are storing.

Later, when we want to use the quantized data, we perform the dequantization process:

```shell
dequantized_v = (quantized_v + 8) * interval + v_min
```
In the dequantization process, we add 8 to the indices to shift the range back from -8-7 to 0-15. Then, we look up the corresponding value in the levels vector to get the original (or close to original) values. This completes the round-trip from quantization to dequantization.


This is a simplified explanation of the process. In practice, the NF4 quantization technique involves other steps, such as bias correction and variance reduction, to ensure that the quantized values accurately represent the original data.

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