---
title: Understanding Async in Python
excerpt: A deep dive into Python's asynchronous programming model - from the event loop to async/await, understanding when and how to write concurrent code
tags: [Python, Async, Concurrency, Programming, asyncio, Event Loop, Coroutines]
date: 2026-01-25 10:30:00 +0530
categories: programming python
toc: true
permalink: /:title
mathjax: false
mermaid: true
---


If you've worked with modern Reinforcement Learning (RL) frameworks, you've probably noticed something: everything is async. Ray's `remote()` calls, distributed training loops, environment rollouts — they all use Python's async primitives. But why? And more importantly, how does async actually work under the hood?

I recently found myself knee-deep in RL training code, staring at `async def`, `await`, and `asyncio.gather()` scattered throughout the codebase. I realized I'd been using these tools without truly understanding the model behind them. When do you use async vs threading vs multiprocessing? Why does the GIL matter for some workloads but not others? What's actually happening when you `await` something?

This post is my attempt to build a solid mental model of Python's concurrency landscape — from the fundamentals of threads and processes, through the constraints imposed by the GIL, to modern async/await patterns. By the end, you'll understand not just *how* to write async code, but *why* RL training pipelines are architected the way they are.

---

## Table of Contents

1. [The Basics: What Are Threads and Processes?](#the-basics-what-are-threads-and-processes)
2. [The Two Types of Waiting](#the-two-types-of-waiting)
3. [Enter the GIL: Python's Original Sin](#enter-the-gil-pythons-original-sin)
4. [Async/Await: Concurrency Without Parallelism](#asyncawait-concurrency-without-parallelism)
5. [The Workarounds We've Lived With](#the-workarounds-weve-lived-with)
6. [Why 896% CPU is Historic](#why-896-cpu-is-historic)
7. [What This Means for You](#what-this-means-for-you)

---

## The Basics: What Are Threads and Processes?

Let's start by understanding the fundamental building blocks of concurrent execution.

### Processes: Separate Worlds

A **process** is an independent program execution with its own memory space. When you open Chrome and Spotify simultaneously, those are separate processes. They can't accidentally overwrite each other's data because they live in completely isolated memory spaces.

<div class="mermaid">
flowchart TB
    subgraph ProcessA["🔷 Process A"]
        MA["Memory<br/>(isolated)"]
        CA["Code"]
    end
    subgraph ProcessB["🔷 Process B"]
        MB["Memory<br/>(isolated)"]
        CB["Code"]
    end
    subgraph Kernel["🖥️ OS Kernel"]
        K[" "]
    end
    ProcessA --> Kernel
    ProcessB --> Kernel
</div>

**Pros:** Complete isolation, true parallelism, crash safety (one process dying doesn't kill others)

**Cons:** Heavy to create (~30MB+ overhead each), expensive communication between processes (serialization/deserialization), no shared memory by default

### Threads: Roommates Sharing an Apartment

A **thread** is a lightweight unit of execution that lives *within* a process. Multiple threads share the same memory space — like roommates sharing an apartment. They can all access the refrigerator (shared memory), which is efficient but dangerous if not coordinated.

<div class="mermaid">
flowchart TB
    subgraph Process["🔷 Process"]
        SM["📦 Shared Memory"]
        T1["🧵 Thread 1"]
        T2["🧵 Thread 2"]
        T3["🧵 Thread 3"]
        SM --> T1
        SM --> T2
        SM --> T3
    end
</div>

**Pros:** Lightweight (~8KB overhead), fast communication (shared memory), quick to spawn

**Cons:** Race conditions, deadlocks, need for synchronization primitives (locks, semaphores)

### The Promise of Multi-Core Systems

Modern CPUs have multiple cores. My laptop has 10 cores. A typical cloud VM might have 64 or 128. The promise is simple: if you have 8 cores and 8 threads doing independent work, you should get ~8x speedup.

**Key insight:** A single core can context-switch between multiple threads (time-slicing), but at any given instant, only **one thread** executes on a core. For true parallelism, you want threads running simultaneously on different cores:

<div class="mermaid">
flowchart TB
    subgraph Core1["⚙️ Core 1"]
        T1["Thread 1<br/>▶️ Work 1"]
    end
    subgraph Core2["⚙️ Core 2"]
        T2["Thread 2<br/>▶️ Work 2"]
    end
    subgraph Core3["⚙️ Core 3"]
        T3["Thread 3<br/>▶️ Work 3"]
    end
    subgraph Core4["⚙️ Core 4"]
        T4["Thread 4<br/>▶️ Work 4"]
    end
    Core1 -.-> Result["✅ All executing simultaneously<br/>(parallel execution)<br/>Total time ≈ Time for 1 task"]
    Core2 -.-> Result
    Core3 -.-> Result
    Core4 -.-> Result
</div>

In C, C++, Java, Go, Rust — this just works. Create threads, distribute work, enjoy parallelism.

In Python? Well...

---

## The Two Types of Waiting

Before we dive into the GIL, we need to understand a crucial distinction that determines which concurrency model you should use.

### I/O-Bound: Waiting for the World

**I/O-bound** tasks spend most of their time waiting for external operations:

- Waiting for a database query to return
- Waiting for an HTTP response from an API
- Waiting for a file to be read from disk
- Waiting for user input

```python
# I/O-bound example
def fetch_user_data(user_id):
    response = requests.get(f"https://api.example.com/users/{user_id}")  # Waiting...
    return response.json()

# If each request takes 100ms, fetching 100 users sequentially = 10 seconds
# But the CPU is idle 99% of that time!
```

The CPU isn't doing work here — it's just waiting. This is like a chef waiting for water to boil. They could be chopping vegetables instead.

### CPU-Bound: The Processor is Sweating

**CPU-bound** tasks keep the processor busy with actual computation:

- Training a neural network
- Computing cryptographic hashes
- Processing images
- Running simulations
- Parsing and transforming large datasets

```python
# CPU-bound example
def compute_hash(data):
    for _ in range(1000000):
        data = hashlib.sha256(data).digest()  # CPU is working hard
    return data
```

Here, the CPU is maxed out. There's no waiting — it's pure computation.

### Why This Distinction Matters

The optimal concurrency strategy depends entirely on which type of work you're doing:

| Task Type | Bottleneck | Solution |
|-----------|------------|----------|
| I/O-Bound | Network, Disk, External Systems | Concurrency (threads, async) |
| CPU-Bound | Processor Speed | Parallelism (multiple cores) |

This brings us to Python's infamous limitation.

---

## Enter the GIL: Python's Original Sin

### The Problem: Reference Counting Isn't Thread-Safe

Python was created in 1991. At that time, most computers had a single CPU core, and multi-threading was rare. The hard problem to solve was memory management.

Python uses **reference counting** for memory management. Every object has an internal counter tracking how many variables (references) point to it. When this counter hits zero, Python knows the object is no longer needed and frees its memory:

```python
a = [1, 2, 3]  # Create a list. Reference count = 1 (only 'a' points to it)
b = a          # 'b' now also points to the SAME list. Reference count = 2
del a          # Remove the 'a' reference. Reference count = 1 (only 'b' remains)
del b          # Remove the 'b' reference. Reference count = 0 → Object is freed!
```

Note: `b = a` doesn't copy the list — both `a` and `b` point to the *same* list object in memory. Python tracks this internally.

**The problem:** Without protection, two threads could modify the reference count simultaneously. Say an object has refcount = 2, and both threads try to add a reference at the same time:

```
Thread 1: reads refcount (2)     Thread 2: reads refcount (2)
Thread 1: computes 2 + 1 = 3     Thread 2: computes 2 + 1 = 3
Thread 1: writes 3               Thread 2: writes 3

Result: 3    Should be: 4    → Reference count is wrong! 💥
```

Now the object might get freed while something still references it — a crash waiting to happen.

### The Solution: The GIL

The **Global Interpreter Lock (GIL)** is a mutex (mutual exclusion lock) that protects access to Python objects. It ensures that **only one thread can execute Python code at any given time**, even on a multi-core machine.

<div class="mermaid">
flowchart TB
    subgraph PythonProcess["🐍 Python Process"]
        GIL["🔒 GIL<br/>(Only ONE thread at a time)"]
        T1["🧵 Thread 1<br/>🏃 I have the GIL,<br/>I can run!"]
        T2["🧵 Thread 2<br/>😴 Waiting..."]
        T3["🧵 Thread 3<br/>😴 Waiting..."]
        GIL --> T1
        GIL -.blocked.-> T2
        GIL -.blocked.-> T3
    end

    subgraph Hardware["💻 Hardware"]
        C1["⚙️ Core 1<br/>BUSY"]
        C2["⚙️ Core 2<br/>IDLE"]
        C3["⚙️ Core 3<br/>IDLE"]
        C4["⚙️ Core 4<br/>IDLE"]
    end

    T1 --> C1
    Note["You have 4 cores, but Python only uses 1.<br/>Max CPU usage: ~100%"]
</div>

The GIL was a simple, elegant solution: just don't let threads run simultaneously. Problem solved... until multi-core CPUs became the norm.

### The GIL's Impact on CPU-Bound Code

Let's see the damage:

```python
import threading
import time

def cpu_intensive_task():
    """Count to 100 million — pure CPU work"""
    count = 0
    for _ in range(100_000_000):
        count += 1
    return count

# Sequential execution
start = time.time()
cpu_intensive_task()
cpu_intensive_task()
sequential_time = time.time() - start
print(f"Sequential: {sequential_time:.2f}s")

# Threaded execution (with GIL)
start = time.time()
t1 = threading.Thread(target=cpu_intensive_task)
t2 = threading.Thread(target=cpu_intensive_task)
t1.start(); t2.start()
t1.join(); t2.join()
threaded_time = time.time() - start
print(f"Threaded: {threaded_time:.2f}s")
```

**Results (CPython with GIL):**
```
Sequential: 6.2s
Threaded: 6.4s  ← Actually SLOWER due to lock contention! 😱
```

The threads aren't running in parallel — they're taking turns, plus paying the overhead of lock acquisition/release. More threads can actually make it **slower**.

### But Wait — Threads Do Help Sometimes!

The GIL is released during I/O operations. When a thread is waiting for network/disk, it releases the GIL, allowing other threads to run:

```python
import threading
import requests
import time

def fetch_url(url):
    """I/O-bound task — network waiting"""
    response = requests.get(url)
    return len(response.content)

urls = ["https://example.com"] * 10

# Sequential
start = time.time()
for url in urls:
    fetch_url(url)
print(f"Sequential: {time.time() - start:.2f}s")

# Threaded
start = time.time()
threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threaded: {time.time() - start:.2f}s")
```

**Results:**
```
Sequential: 5.2s
Threaded: 0.6s  ← ~9x faster! 🎉
```

This works because while Thread 1 is waiting for HTTP response, Thread 2 can grab the GIL and start its request.

<div class="mermaid">
gantt
    title Timeline with GIL (I/O-bound)
    dateFormat X
    axisFormat %s

    section Thread 1
    Request           :t1a, 0, 1
    Waiting for response :t1b, 1, 4
    Process          :t1c, 5, 1

    section Thread 2
    Request          :t2a, 1, 1
    Waiting...       :t2b, 2, 3
    Process         :t2c, 5, 1

    section Thread 3
    Request         :t3a, 2, 1
    Waiting...      :t3b, 3, 3
    Process        :t3c, 6, 1
</div>

*Threads overlap during I/O waits, allowing concurrent execution*

---

## Async/Await: Concurrency Without Parallelism

So threads help with I/O but not CPU-bound work. Python 3.5 introduced another tool: **async/await**.

### The Event Loop Model

Async uses **cooperative multitasking** — a single thread that voluntarily **yields control** (pauses itself) when waiting for I/O, allowing other tasks to run:

```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()  # Yield control while waiting

async def main():
    async with aiohttp.ClientSession() as session:
        # These run concurrently in a SINGLE thread
        tasks = [fetch_url(session, f"https://example.com/{i}") for i in range(100)]
        results = await asyncio.gather(*tasks)
    return results

asyncio.run(main())
```

### What Does `await` Actually Do?

The `await` keyword is the magic that makes async work. It does two things:

1. **Pauses the current coroutine** — "I'm waiting for this result, let others run"
2. **Resumes when ready** — "The result is here, continue from where I left off"

```python
async def example():
    print("Starting request...")
    
    # WITHOUT await - WRONG! This just creates a coroutine object, doesn't run it
    response = fetch_data()  # Returns <coroutine object>, not actual data!
    
    # WITH await - CORRECT! This actually runs the coroutine and waits for result
    response = await fetch_data()  # Pauses here, lets other tasks run, 
                                   # resumes when data arrives
    
    print(f"Got response: {response}")
```

Think of `await` like placing an order at a restaurant:
- **Without `await`**: You hand the waiter a note saying "I want pasta" but walk away before they read it. You never get food.
- **With `await`**: You place your order and wait at your table. While the kitchen cooks, other customers can order too. When your food is ready, the waiter brings it to you.

**Key insight:** `await` is where your coroutine *yields control* back to the event loop. Without `await` points, your async function would block everything else — defeating the purpose of async entirely.

### How the Event Loop Works

The event loop is like a restaurant manager coordinating multiple tables:

<div class="mermaid">
flowchart TB
    subgraph SingleThread["🧵 Single Thread"]
        subgraph EventLoop["⚡ Event Loop"]
            Queue["📋 Task Queue<br/>[Task A] [Task B] [Task C] [Task D] [Task E]"]
            Queue --> Step1["Task A: runs until 'await' → pauses, yields control to event loop"]
            Step1 --> Step2["Task B: runs until 'await' → pauses, yields control"]
            Step2 --> Step3["Task A: I/O complete! resumes from where it paused"]
            Step3 --> Step4["Task C: runs until 'await' → pauses, yields control"]
            Step4 --> Continue["..."]
        end
    end
</div>

**"Yielding control"** means the coroutine voluntarily pauses and tells the event loop: "I'm waiting for something — go run other tasks, and come back to me when my I/O is done."

The key insight: **no parallelism, just efficient scheduling**. While Task A waits for I/O, the event loop runs Task B. No thread switching overhead, no locks needed.

### Async vs Threads: When to Use What?

| Aspect | Threads | Async |
|--------|---------|-------|
| Overhead | ~8KB per thread | ~480 bytes per coroutine |
| Scalability | Thousands | Hundreds of thousands |
| I/O Concurrency | ✅ Good | ✅ Excellent |
| CPU Parallelism | ❌ No (GIL) | ❌ No (single thread) |
| Code Complexity | Moderate | "async everywhere" |
| Existing Libraries | Most work | Need async versions |

### The "What Color is Your Function" Problem

Async introduces a viral constraint — async functions can only be called from other async functions:

```python
async def fetch_data():
    ...

def process_data():
    # ❌ Cannot do this:
    # data = await fetch_data()
    
    # ✅ Must use:
    data = asyncio.run(fetch_data())  # Creates new event loop

# Your entire codebase becomes "colored" — sync or async
```

This has led to parallel ecosystems: `requests` vs `aiohttp`, `psycopg2` vs `asyncpg`, etc.

### Async Doesn't Help CPU-Bound Work Either

```python
import asyncio
import time

async def cpu_task():
    """This blocks the entire event loop!"""
    count = 0
    for _ in range(100_000_000):
        count += 1
    return count

async def main():
    start = time.time()
    # These run SEQUENTIALLY because there's no await inside cpu_task
    await asyncio.gather(cpu_task(), cpu_task())
    print(f"Time: {time.time() - start:.2f}s")

asyncio.run(main())
```

Output: Same as sequential execution. Async is for I/O, not CPU work.

---

## The Workarounds We've Lived With

For 33 years, Python developers have used various workarounds for CPU-bound parallelism.

### 1. Multiprocessing: The Heavyweight Solution

Since threads share the GIL, use separate processes — each gets its own Python interpreter and GIL:

```python
from multiprocessing import Pool
import time

def cpu_intensive_task(n):
    count = 0
    for _ in range(n):
        count += 1
    return count

if __name__ == "__main__":
    start = time.time()
    
    with Pool(processes=4) as pool:
        results = pool.map(cpu_intensive_task, [25_000_000] * 4)
    
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Total: {sum(results)}")
```

**Pros:** True parallelism, uses all cores

**Cons:**
- High memory overhead (each process copies the entire Python interpreter)
- IPC (Inter-Process Communication) requires serialization (pickle)
- Can't share memory easily
- Process creation is slow

<div class="mermaid">
flowchart TB
    subgraph P1["🔷 Process 1"]
        I1["🐍 Python Interpreter<br/>+ GIL #1"]
        M1["💾 Memory: 150MB"]
    end
    subgraph P2["🔷 Process 2"]
        I2["🐍 Python Interpreter<br/>+ GIL #2"]
        M2["💾 Memory: 150MB"]
    end
    subgraph P3["🔷 Process 3"]
        I3["🐍 Python Interpreter<br/>+ GIL #3"]
        M3["💾 Memory: 150MB"]
    end
    P1 <-->|pickle/IPC overhead| P2
    P2 <-->|pickle/IPC overhead| P3
    P1 <-->|pickle/IPC overhead| P3
</div>

### 2. C Extensions That Release the GIL

NumPy, SciPy, and other libraries are written in C and release the GIL during computation:

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def numpy_operation(arr):
    # NumPy releases GIL during this operation!
    return np.fft.fft(arr)

arrays = [np.random.random(1000000) for _ in range(8)]

# This actually runs in parallel because NumPy releases the GIL
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(numpy_operation, arrays))
```

This is why data science in Python "works" — the heavy lifting happens in C, outside the GIL.

### 3. Cython with `nogil`

You can write Python-like code that compiles to C and explicitly releases the GIL:

```cython
# In .pyx file
from cython.parallel import prange

def parallel_sum(double[:] arr):
    cdef double total = 0
    cdef int i
    cdef int n = arr.shape[0]
    
    with nogil:  # Release the GIL!
        for i in prange(n):  # Parallel loop
            total += arr[i]
    
    return total
```

### 4. Numba JIT Compilation

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def parallel_sum(arr):
    total = 0.0
    for i in prange(len(arr)):  # Runs in parallel, no GIL
        total += arr[i]
    return total
```

### The Mental Overhead

Every Python developer has had to internalize this decision tree:

<div class="mermaid">
flowchart TD
    Start["Is my task CPU-bound or I/O-bound?"]

    Start --> IO["I/O-bound"]
    Start --> CPU["CPU-bound"]

    IO --> Few["Few concurrent operations?"]
    IO --> Many["Many concurrent operations?"]
    Few --> Threading1["✅ threading"]
    Many --> Asyncio["✅ asyncio"]

    CPU --> NumPy["Can use NumPy/native libs?"]
    CPU --> PurePython["Pure Python computation?"]
    CPU --> ML["ML training?"]

    NumPy --> Threading2["✅ threading (GIL released)"]

    PurePython --> Rewrite["Can rewrite in Cython/Numba?"]
    PurePython --> MustStay["Must stay pure Python?"]

    Rewrite --> DoThat["✅ Do that"]
    MustStay --> Multiprocessing["✅ multiprocessing"]

    ML --> Framework["✅ Let PyTorch/TensorFlow handle it"]
</div>

This complexity is what made the GIL such a pain point.

---

## Why 896% CPU is Historic

Now we can understand why that screenshot matters.

### What Changed: PEP 703

[PEP 703](https://peps.python.org/pep-0703/) proposed making the GIL optional. After years of work by Sam Gross and others, Python 3.13 shipped with an experimental **free-threaded build** (the `t` in `python3.14t`).

### What 896% CPU Means

<div class="mermaid">
flowchart TB
    subgraph Before["⛔ Before (with GIL) - Max CPU: ~100%"]
        B1["⚙️ Core 1<br/>BUSY"]
        B2["⚙️ Core 2<br/>IDLE"]
        B3["⚙️ Core 3<br/>IDLE"]
        B4["⚙️ Core 4<br/>IDLE"]
    end

    subgraph After["✅ After (free-threaded) - Max CPU: ~896%"]
        A1["⚙️ Core 1<br/>BUSY"]
        A2["⚙️ Core 2<br/>BUSY"]
        A3["⚙️ Core 3<br/>BUSY"]
        A4["⚙️ Core 4<br/>BUSY"]
        A5["... continuing to all 9 cores"]
    end
</div>

For the first time in Python's history, **pure Python threads can execute truly in parallel**.

### Simple Code, Actual Parallelism

```python
# This now actually runs in parallel on free-threaded Python!
import threading

def cpu_work():
    total = 0
    for i in range(100_000_000):
        total += i
    return total

threads = [threading.Thread(target=cpu_work) for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()

# Before: ~6 seconds (sequential, threads fighting for GIL)
# After:  ~0.8 seconds (parallel, all cores utilized)
```

### What Made This Possible?

The implementation required massive changes:

1. **Biased Reference Counting**: Objects start with thread-local reference counts, only switching to atomic operations when shared between threads

2. **Per-Object Locks**: Fine-grained locking replaces the global lock

3. **Deferred Reference Counting**: Some reference count updates are batched

4. **Immortal Objects**: Common objects like `None`, `True`, small integers don't need reference counting at all

5. **Thread-Safe Containers**: `dict`, `list`, etc. now have internal synchronization

---

## What This Means for You

### Short Term (Now - 2026)

The free-threaded build is **experimental**. Don't use it in production yet.

**Current limitations:**
- Many C extensions don't support it yet (NumPy, pandas working on it)
- Some single-threaded code runs ~40% slower due to overhead
- Not all packages are thread-safe

### Medium Term (2026-2028)

As the ecosystem adapts:
- Major libraries will support free-threading
- The performance gap will narrow
- More projects will adopt it for parallel workloads

### Long Term (2028+)

Eventually, the free-threaded build may become the default, and the GIL will be a historical footnote.

### What Should You Do Now?

1. **For I/O-bound work:** Continue using `asyncio` or `threading` — they work great

2. **For CPU-bound work:** 
   - Production: Still use `multiprocessing` or native extensions
   - Experimentation: Try the free-threaded build, report bugs

3. **If you maintain a C extension:** Start testing with free-threaded Python, add the necessary synchronization

4. **For ML/AI workloads:** The impact will be gradual — PyTorch/JAX already handle parallelism at the CUDA level. But free-threading could simplify data loading, preprocessing pipelines, and orchestration code.

---

## Conclusion: The End of an Era

The GIL was a reasonable design choice in 1991. It made Python's memory management simple and safe. But as computing evolved to multi-core parallelism, it became an increasingly painful limitation.

For 33 years, we worked around it with multiprocessing, C extensions, async, and third-party tools. We told ourselves "Python isn't for CPU-bound work" or "just use the right tool for the job."

That screenshot of `python3.14t` at 896% CPU marks the beginning of the end for all those workarounds. Pure Python, using the `threading` module we've had since 1998, can finally use multiple cores.

The GIL is dead. Long live Python.

---

## Quick Reference: Python Concurrency Cheat Sheet

<div class="mermaid">
graph TB
    Title["<b>Python Concurrency Models</b>"]

    subgraph T1["threading (with GIL)"]
        T1A["<b>Best For:</b> I/O-bound tasks"]
        T1B["<b>Mechanism:</b> OS threads, shared memory<br/>GIL limits CPU parallelism"]
    end

    subgraph T2["threading (no-GIL) 🎉"]
        T2A["<b>Best For:</b> I/O AND CPU-bound tasks!"]
        T2B["<b>Mechanism:</b> OS threads, shared memory<br/>True parallelism!"]
    end

    subgraph T3["asyncio"]
        T3A["<b>Best For:</b> High-concurrency I/O"]
        T3B["<b>Mechanism:</b> Single thread, event loop<br/>Cooperative multitasking"]
    end

    subgraph T4["multiprocessing"]
        T4A["<b>Best For:</b> CPU-bound tasks (legacy/stable)"]
        T4B["<b>Mechanism:</b> Separate processes, IPC<br/>Heavy but truly parallel"]
    end

    subgraph T5["C extensions (NumPy etc)"]
        T5A["<b>Best For:</b> Performance-critical compute"]
        T5B["<b>Mechanism:</b> Native code, releases GIL<br/>Best of both worlds"]
    end
</div>

---

*If you found this helpful, you might also enjoy my posts on [RL environments for LLM training](/rl-environments) and [distributed training infrastructure](/distributed-training).*

<!-- # Background

# The Event Loop

# Async/Await Fundamentals

## Coroutines

## Tasks and Futures

# Common Patterns

## Concurrent Operations

## Error Handling

## Timeouts and Cancellation

# When to Use Async

## Async vs Threading vs Multiprocessing

## Performance Considerations

# Real-World Examples

## Web Requests

## Database Operations

## File I/O

# Common Pitfalls

## Blocking the Event Loop

## Mixing Sync and Async Code

## Resource Management

# Best Practices

# Conclusion

# References -->
