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
3. [Enter the GIL: Python's Infamous Lock](#enter-the-gil-pythons-infamous-lock)
4. [Async/Await: Concurrency Without Parallelism](#asyncawait-concurrency-without-parallelism)
5. [Why 896% CPU is Historic](#why-896-cpu-is-historic)
6. [Real-World Example: GRPO Training Loop](#real-world-example-grpo-training-loop)

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

## Enter the GIL: Python's Infamous Lock

<!-- ### The Problem: Reference Counting Isn't Thread-Safe

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

The **Global Interpreter Lock (GIL)** is a mutex (mutual exclusion lock) that protects access to Python objects. It ensures that **only one thread can execute Python code at any given time**, even on a multi-core machine. -->

Python's internals aren't thread-safe i.e. multiple threads modifying the same data structures can corrupt memory. Rather than adding fine-grained locks everywhere (complex and slow), Python uses the **Global Interpreter Lock (GIL)**: a single mutex that ensures **only one thread can execute Python code at any given time**, even on a multi-core machine.

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

**Results (with GIL):**
```
Sequential: 6.2s
Threaded: 6.4s  ← Actually SLOWER due to GIL lock contention!
```

The threads aren't running in parallel — they're taking turns, plus paying the overhead of lock acquisition/release. More threads can actually make it **slower**.

### But Wait — Threads Do Help Sometimes!

The GIL is released during I/O operations. When a thread is waiting for network/disk, it releases the GIL, allowing other threads to run:

```python
import threading
import requests
import time

def simulated_io_task(task_id):
    """Simulate I/O-bound task — sleep releases the GIL"""
    time.sleep(1)  # Simulates waiting for disk/network/database
    return task_id

num_tasks = 5

# Sequential
start = time.time()
for i in range(num_tasks):
    simulated_io_task(i)
print(f"Sequential I/O: {time.time() - start:.2f}s")

# Threaded
start = time.time()
threads = [threading.Thread(target=simulated_io_task, args=(i,)) for i in range(num_tasks)]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threaded I/O: {time.time() - start:.2f}s")
```

**Results:**
```
Sequential: 5.2s
Threaded: 1.01s  -> ~5x speedup!
```

This works because while Thread 1 is waiting for HTTP response, Thread 2 can grab the GIL and start its request.

```
I/O-Bound Threading Timeline (GIL released during waits)
═══════════════════════════════════════════════════════════════════════════════
                    0ms      20ms      40ms      60ms      80ms     100ms
                     │         │         │         │         │         │
Thread 1  ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓
          send                    waiting for response...              done

Thread 2     ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓
             send                 waiting for response...              done

Thread 3        ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓
                send              waiting for response...              done
                     │         │         │         │         │         │
═══════════════════════════════════════════════════════════════════════════════
▓▓▓ = CPU work (has GIL)    ░░░ = Waiting for I/O (GIL released)

💡 All 3 requests sent within ~3ms, all responses arrive ~100ms later
   Total time ≈ 100ms — not 300ms! Threads overlap during I/O waits.
```

*While waiting for I/O, threads release the GIL — other threads can start their requests*

### Why Not Just Use More Threads?

So threads work great for I/O-bound tasks, but they have limits. Each OS thread comes with an overhead: a thread stack and OS-level scheduling/context-switching. For 10 concurrent HTTP requests, threads are fine. But what about 10,000 concurrent connections — streaming data from thousands of RL environments or handling parallel API calls to an LLM provider? A one-thread-per-connection approach can burn gigabytes of thread stack space and spend a lot of time switching between threads instead of doing useful work.

The deeper issue is **how switching happens (preemptive vs cooperative scheduling)**:

- **Threads (preemptive scheduling):** The OS decides when to switch between threads. It can interrupt a thread at *any* point, save its entire state (registers, stack pointer, etc.), and switch to another. This context switch is expensive (~1-10μs) and unpredictable.

- **Async (cooperative scheduling):** Your code decides when to yield control via `await`. No OS involvement, no saving full thread state — just a simple function call to resume a coroutine. Context switch cost: ~100ns (10-100x faster).


## Async/Await: Concurrency Without Parallelism

Python 3.5 introduced **async/await** as a lightweight alternative for high-concurrency I/O.

### The Event Loop Model

Async uses **cooperative multitasking** — sub-tasks (called coroutines) voluntarily **yield control** when waiting for I/O, allowing other coroutines to run. 

**What are coroutines?** They're lightweight Python objects created when you call an `async def` function. Technically, they're a special type of generator — objects that implement the iterator protocol with `__await__`, allowing them to be paused (at `await` points) and resumed. They don't get their own threads; they're just Python objects sitting in memory.

**Async runs entirely on a single thread.** The **event loop** is the scheduler that multiplexes between coroutines — it keeps track of which ones are waiting for I/O and which are ready to run, switching between them whenever one yields:

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

**Key insight:** `await` is where your coroutine *yields control* back to the event loop. Without `await` points, your async function would block everything else — defeating the purpose of async entirely.

### How the Event Loop Works

The event loop maintains a queue of coroutines and runs them one at a time:

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

**"Yielding control"** means the coroutine voluntarily pauses and tells the event loop: "I'm waiting for something — go run 
other tasks, and come back to me when my I/O is done."

When coroutine A hits `await`, it pauses (state saved in the coroutine object), and the event loop picks the next ready coroutine from the queue. When A's I/O completes, it goes back in the queue to be resumed later.

**No parallelism, just efficient scheduling.** While Task A waits for I/O, the event loop runs Task B. No thread switching overhead, no locks needed.

### Visualizing Async Execution

Let's trace through a simple example — making two API calls concurrently:

```python
import asyncio

async def fetch_user():
    print("1. Fetching user...")
    await asyncio.sleep(2)  # Simulate 2-second API call
    print("4. Got user!")
    return {"name": "Alice"}

async def fetch_posts():
    print("2. Fetching posts...")
    await asyncio.sleep(1)  # Simulate 1-second API call
    print("3. Got posts!")
    return [{"title": "Hello"}]

async def main():
    user, posts = await asyncio.gather(fetch_user(), fetch_posts())
    print("5. Done!")

asyncio.run(main())
```

The numbers show execution order. Here's what happens step by step — remember, **everything runs on a single thread**, so only one thing executes at a time:

<div class="mermaid">
flowchart TD
    subgraph T0["⏱️ t=0ms"]
        A1["🔵 <b>fetch_user()</b><br/>print 'Fetching user...'"]:::user
        A2["🔵 await sleep(2) — YIELDS"]:::userpause
        A3["🟠 <b>fetch_posts()</b><br/>print 'Fetching posts...'"]:::posts
        A4["🟠 await sleep(1) — YIELDS"]:::postspause
        A5["⚪ Event loop idle<br/>both waiting..."]:::idle
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph T1["⏱️ t=1000ms — posts timer fires"]
        B1["🟠 <b>fetch_posts() WAKES</b><br/>print 'Got posts!'<br/>✅ done"]:::posts
        B2["⚪ Event loop idle<br/>user has 1s left..."]:::idle
        B1 --> B2
    end

    subgraph T2["⏱️ t=2000ms — user timer fires"]
        C1["🔵 <b>fetch_user() WAKES</b><br/>print 'Got user!'<br/>✅ done"]:::user
        C2["✅ Both done!<br/>print 'Done!'"]:::done
        C1 --> C2
    end

    T0 --> T1 --> T2

    classDef user fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef userpause fill:#93c5fd,stroke:#1e40af,color:#1e3a5f
    classDef posts fill:#f97316,stroke:#c2410c,color:#fff
    classDef postspause fill:#fdba74,stroke:#c2410c,color:#7c2d12
    classDef idle fill:#e5e7eb,stroke:#6b7280,color:#374151
    classDef done fill:#22c55e,stroke:#15803d,color:#fff
</div>

**Output:**
```
1. Fetching user...    ← runs immediately (no await yet)
2. Fetching posts...   ← runs immediately after user yields
3. Got posts!          ← posts timer fires first (1s)
4. Got user!           ← user timer fires second (2s)
5. Done!
```

**Key points:**
- At t=0, both `print()` statements run **synchronously** — no await has happened yet, so no yielding
- `fetch_user()` runs first because it's the first argument to `gather()`
- Only when each task hits `await` does it pause and let the next task run
- Total time = 2 seconds (the slower one), not 3 seconds (1 + 2 if sequential)
- The event loop is **single-threaded** — it runs one thing at a time, but switches between tasks at `await` points

### No `await` = No Concurrency

If there's no `await`, async functions run **purely sequentially** — the `async` keyword alone does nothing for concurrency:

```python
import asyncio
import time

async def task_a():
    print("A: start")
    time.sleep(1)  # Regular sleep — BLOCKS everything!
    print("A: end")

async def task_b():
    print("B: start")
    time.sleep(1)  # Regular sleep — BLOCKS everything!
    print("B: end")

async def main():
    await asyncio.gather(task_a(), task_b())

asyncio.run(main())
```

**Output (takes 2 seconds!):**
```
A: start
A: end      ← A runs completely before B even starts
B: start
B: end
```

<div class="mermaid">
flowchart LR
    A1["🔵 A: start"]:::taskA --> A2["🔵 sleep(1)<br/>🚫 BLOCKS"]:::taskAblock --> A3["🔵 A: end"]:::taskA --> B1["🟠 B: start"]:::taskB --> B2["🟠 sleep(1)<br/>🚫 BLOCKS"]:::taskBblock --> B3["🟠 B: end"]:::taskB

    classDef taskA fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef taskAblock fill:#93c5fd,stroke:#1e40af,color:#1e3a5f
    classDef taskB fill:#f97316,stroke:#c2410c,color:#fff
    classDef taskBblock fill:#fdba74,stroke:#c2410c,color:#7c2d12
</div>

Compare with `await asyncio.sleep()`:

```python
async def task_a():
    print("A: start")
    await asyncio.sleep(1)  # Yields control!
    print("A: end")

async def task_b():
    print("B: start")
    await asyncio.sleep(1)  # Yields control!
    print("B: end")
```

**Output (takes 1 second!):**
```
A: start
B: start    ← B starts while A is waiting
A: end
B: end
```

<div class="mermaid">
flowchart TD
    subgraph Concurrent["With await — 1 second total"]
        C1["🔵 A: start"]:::taskA --> C2["🔵 await sleep(1)<br/>💤 yields"]:::taskApause
        C2 --> C3["🟠 B: start"]:::taskB --> C4["🟠 await sleep(1)<br/>💤 yields"]:::taskBpause
        C4 --> C5["⚪ ...1 second passes..."]:::idle
        C5 --> C6["🔵 A: end"]:::taskA --> C7["🟠 B: end"]:::taskB
    end

    classDef taskA fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef taskApause fill:#93c5fd,stroke:#1e40af,color:#1e3a5f
    classDef taskB fill:#f97316,stroke:#c2410c,color:#fff
    classDef taskBpause fill:#fdba74,stroke:#c2410c,color:#7c2d12
    classDef idle fill:#e5e7eb,stroke:#6b7280,color:#374151
</div>

**The rule:** `await` is the yield point. No `await` = no opportunity for other tasks to run.

<!-- ### Async vs Threads: When to Use What?

| Aspect | Threads | Async |
|--------|---------|-------|
| Overhead | ~8KB per thread | ~480 bytes per coroutine |
| Scalability | Thousands | Hundreds of thousands |
| I/O Concurrency | ✅ Good | ✅ Excellent |
| CPU Parallelism | ❌ No (GIL) | ❌ No (single thread) |
| Code Complexity | Moderate | "async everywhere" |
| Existing Libraries | Most work | Need async versions | -->

### Common Async Mistakes That Kill Performance

Async code looks simple, but there are several ways to accidentally destroy your concurrency. Here are the most common pitfalls:

#### 1. Blocking the Event Loop with CPU Work

```python
import asyncio

async def process_image(data):
    # ❌ This blocks the ENTIRE event loop!
    # No other coroutines can run during this computation
    result = heavy_image_processing(data)  # CPU-bound, no await
    return result

async def main():
    # These run SEQUENTIALLY, not concurrently!
    await asyncio.gather(
        process_image(img1),
        process_image(img2),
        process_image(img3),
    )
```

**The fix:** Offload CPU work to a thread pool:

```python
async def process_image(data):
    loop = asyncio.get_event_loop()
    # ✅ Run CPU work in a thread, freeing the event loop
    result = await loop.run_in_executor(None, heavy_image_processing, data)
    return result
```

#### 2. Sequential `await` When You Want Concurrency

```python
async def fetch_all_data():
    # ❌ These run one after another — 3 seconds total
    user = await fetch_user()      # 1 second
    posts = await fetch_posts()    # 1 second  
    comments = await fetch_comments()  # 1 second
    return user, posts, comments

async def fetch_all_data():
    # ✅ These run concurrently — 1 second total
    user, posts, comments = await asyncio.gather(
        fetch_user(),
        fetch_posts(),
        fetch_comments(),
    )
    return user, posts, comments
```

#### 3. Using Blocking I/O Libraries

```python
import requests  # Synchronous library!

async def fetch_url(url):
    # ❌ requests.get() blocks the entire event loop
    response = requests.get(url)
    return response.json()
```

**The fix:** Use async-native libraries:

```python
import aiohttp

async def fetch_url(url):
    # ✅ aiohttp properly yields control during I/O
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

#### 4. Creating Too Many Concurrent Connections

```python
async def fetch_all(urls):
    # ❌ 10,000 simultaneous connections = angry servers, rate limits, crashes
    return await asyncio.gather(*[fetch(url) for url in urls])
```

**The fix:** Use a semaphore to limit concurrency:

```python
async def fetch_all(urls, max_concurrent=100):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_limited(url):
        async with semaphore:
            return await fetch(url)
    
    # ✅ At most 100 concurrent requests
    return await asyncio.gather(*[fetch_limited(url) for url in urls])
```

#### 5. Forgetting to `await`

```python
async def save_to_db(data):
    await db.insert(data)

async def handler(request):
    data = parse_request(request)
    save_to_db(data)  # ❌ Missing await! Returns a coroutine object, never executes
    return {"status": "saved"}  # False signal. Nothing was saved
```

Python emits a `RuntimeWarning: coroutine 'save_to_db' was never awaited` — but only at garbage collection time, not when the bug occurs. In noisy logs or production environments, this warning is easy to miss. Your function returns successfully, the response looks correct, but the database write never happened.

**The golden rule:** Every long-running operation inside an async function needs an `await`. If there's no `await`, there's no concurrency — you're just writing complicated synchronous code.


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

<!-- ---

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

--- -->

## Real-World Example: GRPO Training Loop

Let's look at a real async training loop from [TorchForge](https://github.com/meta-pytorch/torchforge/) — a distributed RL framework. This is the main GRPO (Group Relative Policy Optimization) training script, and it's a perfect example of why async shines for orchestrating distributed ML workloads.

The architecture is simple: **32 rollout coroutines** generate training data by calling remote services (dataloader, LLM generator, reward model), while **1 training coroutine** consumes from a shared replay buffer. All 33 coroutines run on a single thread, coordinated by the event loop.

### The Rollout Coroutine

Each rollout coroutine spends most of its time waiting for remote services:

```python
async def continuous_rollouts():
    while not shutdown_event.is_set():
        # 1. Sample from dataloader (I/O - await)
        sample = await dataloader.sample.call_one()
        
        # 2. Generate responses from LLM (I/O - await, ~seconds)
        responses = await generator.generate.route(prompt)
        
        # 3. Compute rewards (I/O - await)
        reward = await reward_actor.evaluate_response.route(...)
        
        # 4. Get reference logprobs (I/O - await)
        ref_logprobs = await ref_model.forward.route(input_ids)
        
        # 5. Compute advantages and add to buffer (I/O - await)
        advantages = await compute_advantages.compute.call_one(episodes)
        await replay_buffer.add.call_one(episode)
```

**Every `await` is a yield point.** While Rollout 1 waits for the generator, Rollouts 2-32 can make progress. This is I/O-bound concurrency — the CPU isn't doing heavy work; it's orchestrating remote calls.

### See It In Action

Click "Step" to watch the event loop switch between coroutines at each `await`:

<iframe src="/assets/visualizations/grpo-async-flow.html" width="100%" height="700" style="border: none; border-radius: 12px; margin: 20px 0;"></iframe>

### The Training Coroutine

Meanwhile, a single training coroutine consumes from the replay buffer:

```python
async def continuous_training():
    while training_step < max_steps:
        batch = await replay_buffer.sample.call_one()
        if batch is None:
            await asyncio.sleep(0.1)  # Buffer empty — yield, let rollouts fill it
        else:
            await trainer.train_step.call(batch)
            await trainer.push_weights.call()
            await generator.update_weights.fanout()
```

### Putting It Together

```python
# Launch 32 rollout coroutines + 1 training coroutine
rollout_tasks = [asyncio.create_task(continuous_rollouts()) for _ in range(32)]
training_task = asyncio.create_task(continuous_training())

await training_task  # Run until training completes
```

**The result:** 32 concurrent rollouts, all making progress, all on a single thread. No GIL contention, no thread synchronization, no race conditions. The event loop efficiently multiplexes between coroutines at each `await` point.

---

## Conclusion

The GIL was a reasonable design choice in 1991, but it became a painful limitation as multi-core CPUs became the norm. For decades, we worked around it with multiprocessing, C extensions, and async.

Python 3.13+'s free-threaded build changes everything: pure Python threads can finally use multiple cores. For RL workloads, this means simpler code for parallel environment rollouts, data preprocessing, and orchestration — without the overhead of multiprocessing or the complexity of async everywhere.

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
