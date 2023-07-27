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

End credits already? But, what if you have 1000s of customers? You can't deploy those 1000 clones of those GPU hungry LLMs. Unless you have a Gilfoyle in your team, you can't afford to do that.

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

Now, if W is 10000 x 10000, it means $\Delta W$ is also 10000 x 10000. This is a lot of memory. This is where LoRA comes into the picture. LoRA is a technique to reduce the memory footprint of $\Delta W$. It does this by using a low-rank approximation of $\Delta W$. This is done by decomposing $\Delta W$ into two matrices $UW_{a}$ and $W_{b}$.

Let's break down everything step-by-step:

* If I have a matrix of size 2 x 2, it means 4 elements are stored in the memory. If the matrix is 100 x 100, it means alot of elements are stored in memory ($10000$). What if we there's a better way to store the same information?? Here comes the SVD,

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
input_dim = 100  # e.g., the hidden size of the pre-trained model (W = __100__ x 100)
output_dim = 100  # e.g., the output size of the layer (W = 100 x __100__)
rank = 3  # The rank 'r' for the low-rank adaptation

W = 100x100 matrix # from pretrained network with shape input_dim x output_dim

W_A = nn.Parameter(torch.empty(input_dim, rank)) # LoRA weight A
W_B = nn.Parameter(torch.empty(rank, output_dim)) # LoRA weight B

# Initialization of LoRA weights
nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
nn.init.zeros_(W_B)

def regular_forward_matmul(x, W):
    h = x @ W
return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # regular matrix multiplication
    h += x @ (W_A @ W_B) # updated equation
return h
```

Did you see what we did here? We added $x @ (W_A @ W_B)$ to the existing equation. Since we are freezing W, the only thing that needs gradient updates are $(W_A \& W_B)$. The final weights $ W_{a} \times W_{b} $ are the delta weights $\Delta W$ we need for our finetuned task.

## QLoRA

Although you can store the finetuned weights of a 33B model in the disk into the disk, you would still need a big GPU to load the 33B model into the memory to perform LoRa training. You would have to be rich to save money. Bwahaha

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/gbelson.jpg"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/llora_blog/gbelson.jpg"></a>
    <figcaption><b>Figure 1:</b> <i>Let's get those hotcakes called LLMs</i></figcaption>
</figure>

Worry not. QLoRa to the rescue.