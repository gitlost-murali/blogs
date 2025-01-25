---
title: "CUDA Studylog #1 - Getting a taste of CUDA kernels"
excerpt: "A Introduction Guide for ML Engineers. Learn the fundamentals and practical implementations needed to get started with CUDA kernels"
tags: [CUDA, GPU, ML, Optimization, Performance]
date: 2025-01-25 05:28:10 +0530
categories: machine-learning data-science
toc: true
permalink: /:categories/:title
---

Consider this: Training a large language model can cost upwards of \\$10 million in compute resources. In this context, a seemingly modest 5% improvement in GPU utilization through optimized CUDA kernels could translate to \\$0.5 million in savings. This is why understanding CUDA programming isn't just a technical skill—it's a strategic advantage. So, this blogpost is designed to demystify CUDA programming by focusing on fundamental concepts and practical implementation.

After reading this blogpost, you will understand:
- How GPU parallelization differs from CPU processing
- How to think about your ML problems in terms of parallel operations
- How to convert a simple Python operation into a CUDA kernel
- The basic building blocks of CUDA programming (threads, blocks, grids)
## Why Another CUDA Tutorial?

As an ML engineer diving into CUDA, I found myself asking questions that weren't addressed in standard tutorials, including the excellent ones from [gpu-mode](https://github.com/gpu-mode/lectures). This guide aims to fill that gap, focusing on building intuition and making the journey into CUDA kernel programming less intimidating. We'll work through a simple example that demonstrates the key concepts you need to know. While reading this blog, I suggest you open this [notebook](https://github.com/gpu-mode/lectures/blob/main/lecture_003/pmpp.ipynb) in google colab and play with it as we go.

## Understanding GPU Architecture

Think of CPUs and GPUs as two different specialists working together. Your CPU excels at complex sequential tasks, like a highly skilled individual worker. In contrast, a GPU shines when performing the same operation thousands or millions of times simultaneously, like having an army of workers each doing one simple task. For instance, a modern GPU has over 2³⁰ cores, making it perfect for parallel processing. Neither approach is inherently better—they're suited for different challenges.

### Key Components and Concepts

When we write CUDA code, we're essentially orchestrating several key components:

1. Kernel: Despite its complex-sounding name, __a kernel is simply a function that runs on the GPU__. When we "launch a kernel," we're telling the GPU, "Here's the program; now run it on many threads in parallel." The key difference from regular functions is that a kernel executes across many threads simultaneously.
2. Thread: A thread is the smallest unit of execution in GPU programming. Think of it as a single worker that can perform one set of instructions. Each thread runs the same program (our kernel) but typically works on different data.
3. Memory Hierarchy:
   - Global Memory: The GPU's main memory, accessible by all threads but relatively slow. This is the 40GB/80GB VRAM mentions you see everywhere.
   - Shared Memory: Fast memory shared between threads in the same block
   - Registers: The fastest memory, private to each thread
   - L1/L2 Cache: Automatic caching layers that help speed up memory access

### Task and Data Parallelism

Let's consider a simple example of how GPUs leverage parallelism. Suppose you have two independent operations:
   - Multiplying numbers a and b
   - Adding numbers c and d

On a CPU, these operations would typically happen one after the other. In contrast, a GPU can assign separate threads to handle each operation simultaneously, enabling parallel execution. This capability is what gives GPUs their incredible performance potential.

But here's the key: The performance gains from GPU computing depend entirely on your ability to identify which parts of your program can run in parallel. Not all problems can be parallelized effectively, and sometimes the overhead of moving data between CPU and GPU can outweigh the benefits of parallel execution.

The key to understanding CUDA programming is to **adopt an output-first mindset**. Start by focusing on the desired output, and then map it to the GPU’s computational model. By organizing your kernel execution around the expected output, you can more easily design a grid of thread blocks and achieve efficient parallelism.

Here's why this matters: Modern GPUs can handle up to 1,024 threads per block and more than 2³⁰ blocks in total. That's an enormous amount of parallel computing power. But how do you harness it effectively? Let's explore through it through the following example:

## Practial Implementation: Image Grayscale Conversion (1D Grid)

Let's start with a simple but practical example: converting an RGB image to grayscale. This is a perfect introduction to CUDA because each output pixel can be computed independently.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/cuda-intro/rgb2gray.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/cuda-intro/rgb2gray.png"></a>
    <figcaption><b>Figure 1:</b> <i>RGB to Grayscale conversion</i></figcaption>
</figure>

Let's use OpenCV's grayscale conversion formula and

$$
Gray-pixel =  (0.299 × Red) + (0.587 × Green) + (0.114 × Blue)
$$

Write it in Python to understand the computation at pixel level:

```python
def rgb_to_grayscale(pixel):
    # Assuming pixel is numpy array with shape (3) -> for 3 RGB channels
    red, green, blue = pixel[0], pixel[1], pixel[2]
    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
    return grayscale
```

Let's extend this function to all the pixels in the image:

```python

# input_image's shape => [3, height, width] # 3 for RGB channels
# output_image's shape => [height, width] # since the 3 channels are merged into one now

for row_idx in height:
	for col_idx in width:
		output_image[row_ix, col_ix] = rgb_to_grayscale( input_image[row_idx, col_idx] )
```

Since each color pixel computation to grayscale is independent, we can parallelize this function i.e call this function multiple times with different pixels as inputs. Before jumping to CUDA, let's rewrite our Python code to mirror how CUDA thinks. Two key things to understand:

> 1. CUDA __does not natively support multi-dimensional arrays__ in the same way as standard C/C++. Instead, __a multi-dimensional array is often represented as a flattened 1D__ array in memory. You calculate the index using the formula $index=i×width+j$, where i and j are row and column indices, respectively

> 2. **A kernel can not return anything. It can only change contents of things passed to it.**

Here's our Python code rewritten to match these constraints:
```python

def rgb2grey_pixel_level_kernel(pixel_index, output_grey_tensor, input_flatten_tensor, num_pixels_per_channel) -> None:

output_grey_tensor[pixel_index] = 0.2989*input_flatten_tensor[pixel_index] + \

0.5870*input_flatten_tensor[pixel_index + num_pixels_per_channel] + \

0.1140*input_flatten_tensor[pixel_index + (2*num_pixels_per_channel)]

```

Since each thread would handle one pixel's conversion, let's call the conversion kernel for all pixels in the expected output: 
```python
def run_kernel_manytimes(func, num_times, *args):
	for indx in range(num_times): func(indx, *args)
	
def launch_rgb2grey(input_image):
	c, h, w = input_image.shape
	num_pixels_per_channel = h * w
	
	flattened_input = input_image.flatten() # [h,w,3] => 1D of shape [h x w x 3]
	expected_output = torch.empty(h * w)
	
	number_of_kernel_calls_to_make = len(expected_output)
	
	run_kernel_manytimes(rgb2grey_pixel_level_kernel, \ # kernel to be called 
				number_of_kernel_calls_to_make,  \ # number of times to call it
				output_grey_tensor, input_image, num_pixels_per_channel)#args
	return output_grey_tensor.view(h, w)
```

This is great. If the image shape is  [3, 1280, 720], the number of parallel threads needed are 1280 x 720 since the output is a 2d [1280, 720]. In reality, we can't launch millions of threads simultaneously. CUDA organizes threads into blocks, with a maximum of 1,024 threads per block on modern GPUs.

### Blocks and Grids: The Building Blocks of Parallelism

While threads are powerful, CUDA organizes them into larger structures for better management and scalability:

1. **Blocks**: A block is a group of threads that can work together. Threads within a block can:
    - Share memory resources
    - Synchronize their execution
    - Cooperate on data processing
2. **Grids**: A grid is a collection of blocks. This two-level hierarchy allows CUDA programs to scale across different GPU architectures.

```
Grid
|----Blocks
|----------Threads
```

Since there are limited threads per block (1,024 on modern GPUs), we need to organize our parallel computation carefully. Let's think about this step by step for our grayscale conversion example.

### From Threads to Blocks: A Practical Approach

When converting our 1280x720 image, we need 921,600 threads (1280 * 720). Since we can only have 1,024 threads per block, we need to split this work across multiple blocks. Here's how we can think about it:
<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/cuda-intro/block-thread.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/cuda-intro/block-thread.png"></a>
    <figcaption><b>Figure 2:</b> <i>Block and Threads <a href="http://gpu.di.unimi.it/books/PMPP-3rd-Edition.pdf"> (Image Source)</a> </i></figcaption>
</figure>

1. First, let's calculate how many blocks we need:
```python
threads_per_block = 1024  # Maximum threads we can have per block
total_threads_needed = height * width  # 1280 * 720 = 921,600
num_blocks = (total_threads_needed + threads_per_block - 1) // threads_per_block
```
The formula `(total_threads_needed + threads_per_block - 1) // threads_per_block` might look complex, but it's just ceiling division. For our image:
```python
num_blocks = (921,600 + 1024 - 1) // 1024 = 901 blocks
```
### Understanding Thread/Block Indexing

Now comes the crucial part: how does each thread know which pixel to process? In CUDA, each thread can identify itself using two pieces of information:

- `threadIdx.x`: Its position within its block (0 to 1023 in our case)
- `blockIdx.x`: Which block it belongs to (0 to 900 in our case)

We can calculate the global pixel index that each thread should process:
```python
global_thread_id = blockIdx.x * blockDim.x + threadIdx.x
```
Where:

- `blockDim.x` is the number of threads per block (1024 in our case)
- `threadIdx.x` is the thread's position within its block
- `blockIdx.x` is the block number

> Note: Ignore the `.x` for now. We will discuss it a bit later [Todo: integrate smoothly and explain]

What this means is that, instead of passing thread index directly, we pass `blockIdx` and
`threadIdx` to figure out the global thread idx or pixel idx

In a scenario where the total pixels are 514 and blockDim is 256. In this case, ceiling division would give us 3 blocks. In the 3rd block, we only want to use 2 threads for the 2 pixels as the remaining 512 pixels are taken care of the first two blocks. So, we keep a `if` condition to avoid this overflow. If the thread id is more than the number of pixels, we avoid the computation.


```python
def rgb2grey_pixel_level_kernel(blockidx, threadidx, blockDim, output_grey_tensor, input_flatten_tensor, num_pixels_per_channel) -> None:

global_thread_pixel_id = blockIdx * blockDim + threadIdx

if global_thread_pixel_id < num_pixels_per_channel:
	output_grey_tensor[global_thread_pixel_id] = 0.2989*input_flatten_tensor[global_thread_pixel_id] + \
	0.5870*input_flatten_tensor[global_thread_pixel_id + num_pixels_per_channel] + \
	0.1140*input_flatten_tensor[global_thread_pixel_id + (2*num_pixels_per_channel)]

```
This changes the `for loop` that iterates the function calls. We will have 2 `for loops` now. One for iterating blocks and the other for iterating threads inside each block:

```python
def rgb2grey_block_level_kernel(func, num_blocks, num_threads_per_block, *args):
	for block_idx in range(num_blocks):
		for thread_idx in range(threads): func(block_idx, thread_idx, num_threads_per_block, *args)

def rgb2grey_py_kernel(input_image):
	c, h, w = input_image.shape
	num_pixels_per_channel = h * w

	input_image = input_image.flatten()
	output_grey_tensor = torch.empty(h * w, dtype = input_image.dtype, device = input_image.device)
		

	num_threads_per_block = 256
	num_blocks = int(math.ceil(len(output_grey_tensor)/num_threads_per_block))

	
	run_many_kernels_block(pixel_operation_kernel_block, num_blocks,
	num_threads_per_block, output_grey_tensor, input_image, num_pixels_per_channel)
	
	return output_grey_tensor.view(h, w)
```

Now that we understand how to organize our computation with blocks and threads in Python, let's translate this to actual CUDA code. __The key difference is that instead of explicitly running loops over blocks and threads, CUDA will handle this parallelization for us.__

### CUDA Implementation (1D Grid)

Here's how we implement our grayscale conversion in CUDA:
```cpp
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {

int i = blockIdx.x*blockDim.x + threadIdx.x;

if (i<n) out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n];

}


torch::Tensor rgb_to_grayscale(torch::Tensor input) {
	
	int h = input.size(1);
	
	int w = input.size(2);
	
	printf("h*w: %d*%d\n", h, w);
	
	auto output = torch::empty({h,w}, input.options());
	
	int threads = 64;
	
	rgb_to_grayscale_kernel<<<cdiv(w*h,threads), threads>>>(
	
	input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
	
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	
	return output;
```

Let's break down the key differences from our Python implementation:

1. `__global__` keyword: This tells CUDA that this is a kernel function that can be called from CPU code and runs on the GPU.
2. No explicit loops: Instead of our Python implementation's nested loops, CUDA handles thread creation and management.
3. Boundary check: We add `if (tid < width * height)` to ensure we don't process beyond our image boundaries.
4. The `<<<num_blocks, threads_per_block>>>` syntax is CUDA's way of specifying the grid and block dimensions. This replaces our Python implementation's explicit loops over blocks and threads.


## Next Steps

This implementation serves as a foundation for understanding CUDA programming. In the next post, we'll explore:
- 2D grid implementations for the grayscale conversion and matrix operations
- Using shared memory to reduce global memory access
- Coalesced memory access patterns
- Bank conflicts and how to avoid them
- Warp-level programming
