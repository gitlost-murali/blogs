---
title: Building RL Environments for LLM Training - From Car Racing to Code Agents
excerpt: A deep dive into designing environments for reinforcement learning with LLMs - understanding actions, observations, rewards, and scaling challenges
tags: [Machine Learning, Language Models, Reinforcement Learning, Environments, Agents, GRPO, Training]
date: 2025-12-20 10:30:00 +0530
categories: machine-learning data-science
toc: true
permalink: /:categories/:title
mathjax: true
---

# Background

Reinforcement learning works on the FAFO principle → Fool Around and Find Out ([more on this here]({{ site.url }}/{{ site.baseurl }}/grpo-intro/)). But to fool around, LLMs need a playground: an *environment* where they can take actions, observe outcomes, and learn from their mistakes. 

<!-- If you've ever played a video game, you already grasp the core idea of RL environments.  -->
Imagine a car racing game: Your keyboard inputs (up/down/left/right) go into the game engine, the game world executes a step and returns an outcome: 

1. Did you crash, 
2. Did you successfully cross the finish line, or 
3. Are you still racing? 

This ***action → outcome loop*** is exactly what RL environments provide for LLM training.


<figure style="max-width: 400px; margin: 0 auto;">
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/environments/car_arrows.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/environments/car_arrows.png" style="width: 100%; height: auto;"></a>
    <figcaption><b>Figure 1:</b> <i>A car racing game illustrating the RL loop: actions (arrows for up/left/right) lead to observations (track state) and rewards (progress towards finish line)</i></figcaption>
</figure>


This blog is an attempt to synthesize findings from the existing literature and blogs online.

<!-- Despite being widely used in training reasoning models like DeepSeek-R1, Qwen and Kimi, the infrastructure and environment design details remain poorly documented.  -->

 <!-- In this blog, we will cover the following topics: verification strategies, reward engineering, curriculum learning, sandboxing at scale, and the failure modes that derail training. -->


# The Anatomy of an Environment

<!-- At its core, an RL environment is a state machine that responds to an agent’s actions with new observations and rewards.  -->
Whether you're training a robot to walk or an LLM to write code, the core interface remains the same (mostly):

```python
class Environment:
    def reset(self) -> Observation:
        """Reset the environment to initial state"""
        pass
    
    def step(self, action: Action) -> Tuple[Observation, Reward, Done, Info]:
        """Take an action and return the outcome"""
        pass
```
<!-- 
In our car racing analogy:

- **Action**: Your keyboard input (up/down/left/right)
- **Observation**: The current game state (car position, track layout, other cars)
- **Reward**: Points for moving forward, penalty for hitting walls
- **Done**: Whether the race is over (finished, crashed, or timed out)
- **Info**: Additional metadata (lap time, fuel remaining)

For LLM environments:
- **Action**: The model's generated text response
- **Observation**: Tool outputs, error messages, test results, environment feedback
- **Reward**: Score from verifier (correctness, helpfulness, etc.)
- **Done**: Whether to stop or continue interacting with the environment (task complete, max turns reached, etc.)
- **Info**: Execution traces, intermediate states -->

In our car racing analogy versus LLM training, the parallel is as follows:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              RL ENVIRONMENT                                │
├────────────────────────────┬───────────────────────────────────────────────┤
│         CAR RACING         │               LLM TRAINING                    │
├────────────────────────────┼───────────────────────────────────────────────┤
│ Action: ↑ ↓ ← →            │ Action: Generated text/code/toolcalls         │
│ Observation: Track state   │ Observation: Tool outputs, errors, feedback   │
│ Reward: +1 forward         │ Reward: Verifier score                        │
│ Done: Finish/Crash         │ Done: Task complete / max turns               │
│ Info: Lap time             │ Info: Execution trace                         │
└────────────────────────────┴───────────────────────────────────────────────┘
```

The critical difference lies in the **Reward**. In a game, the game engine knows the score. In LLM training, respective environment would carry out the reward logic. This brings us to the first concept: Verification.

# Reward Verification strategies

Every environment needs to answer two questions: 
1. ***"Did the model do it right?"*** 
2. ***"Should we keep going?"***. 

In other words, we need a *verifier* to compute the reward and a *criterion* to decide if the task is done. 


The choice of verification approach fundamentally shapes what behaviors RL can reinforce. For math, this is often simple: extract the number and compare it to a ground truth. But for code, "correctness" is a spectrum. We can't just string-match code because there are infinite ways to write the same function. 

<!-- making training more sample-efficient and enabling smarter inference-time regeneration strategies. -->

Code verification strategies have converged to five main approaches, each with distinct tradeoffs that affect training dynamics. 

### 1. Execution-Only Verification

The most lenient check: just ensure the code runs without crashing. This is useful for open-ended creative tasks or assigning partial credit for syntactically correct code. 

```python
def verify_runs(code: str) -> float:
    result = sandbox.run(code, timeout=5)
    return 1.0 if result.exit_code == 0 else 0.0
```

For instance, [CodeRL (Le et al., 2022)](https://arxiv.org/abs/2207.01780) treated code generation as an RL problem with execution-based rewards, using a critic network trained to predict functional correctness from four outcome categories: __compile error__, runtime error, failed tests, and passed tests.
<!-- The key innovation was that once trained, the critic could provide dense reward estimates during generation, complementing the sparse rewards from actual execution. -->

$$
r(W_s) = \begin{cases} 
-1.0 & \text{if } W_s \text{ cannot be compiled (compile error)} \\[0.5em]
-0.6 & \text{if } W_s \text{ cannot be executed (runtime error)} \\[0.5em]
-0.3 & \text{if } W_s \text{ failed any unit test} \\[0.5em]
+1.0 & \text{if } W_s \text{ passed all unit tests}
\end{cases}
$$

We can see that instead of a sparse binary reward, we can get -1.0 (worst case: not compiled) or -0.6 (compiled but not executed), -0.3 (executed but failed any unit test), +1.0 (passed all unit tests) based on the outcome. This is a more informative reward signal that can help the model learn from its mistakes.

### 2. Input/Output Matching
Run the code and compare output against expected results. This can be done via stdin/stdout (language-agnostic but fragile to whitespace) or by calling the function directly with arguments. Seen in benchmarks like **LiveCodeBench**, **APPS**, and **CodeContests**.

```python
def verify_stdin_stdout(code: str, test_case: TestCase) -> bool:
    result = sandbox.run(code, stdin=test_case.input)
    return result.stdout.strip() == test_case.expected_output.strip()

def verify_functional(code: str, test_case: TestCase) -> bool:
    actual = sandbox.call_function(code, test_case.func_name, test_case.input)
    return actual == test_case.expected_output
```

### 3. Assertion-Based Testing
Here, we wrap the solution in a test harness with `assert` statements (or unit tests). If the test script exits with code 0, the solution is correct. Seen in benchmarks like **HumanEval**, **MBPP**, etc. 

>**Note:** [EvalPlus](https://github.com/evalplus/evalplus) extended these datasets by generating **80x more test cases** for HumanEval and **35x more** for MBPP using automated input generation seeded by commercial LLMs.

<!-- The pass@k metric—probability that at least one of k samples passes all assertions—has become the standard evaluation paradigm. -->

```python
def verify_with_assertions(solution_code: str, test_code: str) -> bool:
    # assert candidate(input) == expected_output
    # test_code contains: assert candidate([1,2,3]) == 6
    full_code = f"{solution_code}\n\n{test_code}"
    result = sandbox.run(full_code)
    return result.exit_code == 0
```

### 4. Bidirectional verification

What makes Bidirectional verification interesting is that instead of optimizing just for code correctness, it is possible to **optimize for both code correctness and unit test correctness in one go**. [CURE (Yin jie et al., 2025)](https://arxiv.org/abs/2509.14436) proposes co-evolving a coder and unit tester within a single policy (a.k.a LLM). 

For each task, the model generates *n* code solutions and *m* unit tests, then executes all codes against all tests (ground-truth tests and the generated unit tests) to build a binary pass/fail matrix **B***. Code rewards are simply the number of ground-truth tests passed. The clever part is the unit test reward: **+1** for correct behavior (passing correct code, failing incorrect code), **−1** for incorrect behavior (failing correct code, passing incorrect code) - where code "correctness" is determined by ground-truth tests. Both reward signals are normalized and fed into GRPO to update the shared policy.

**💡 Intuition:** A good unit test gets positive reward when it: (1) passes ALL correct code solutions, AND (2) fails as many incorrect code solutions as possible. A bad unit test gets negative reward when it: fails correct code solutions OR passes too many incorrect code solutions.
{: .notice--info}

<iframe 
  src="{{ site.url }}/{{ site.baseurl }}/assets/visualizations/rl_envs/curepipeline.html" 
  style="width: 100%; height: 660px; border: none; border-radius: 16px; margin: 24px 0;"
  loading="lazy"
  title="CURE Pipeline Interactive Visualization">
</iframe>

<!-- A production-grade **Code Environment** combines these into a single `step` method. It takes the model's code (Action), runs the verification suite, and returns the results.

```python
class SingleTurnCodeEnv(Environment):
    def __init__(self, problem: str, test_cases: List[TestCase]):
        self.problem = problem
        self.test_cases = test_cases
    
    def step(self, llm_code: str) -> Tuple[str, float, bool, dict]:
        results = []
        for tc in self.test_cases:
            # Run the chosen verification strategy
            passed = self.sandbox.run_with_assertions(llm_code, tc.assertions)
            results.append(passed)
        
        # Calculate dense reward (percentage of tests passed)
        reward = sum(results) / len(results)
        return "", reward, True, {"passed": sum(results)}
``` -->

# From LLMs to Agents

In early experiments, an LLM's "environment" might be just a single-turn task: get a prompt, output code, get a reward. But real-world tasks are rarely one-turn. An agent might need to *ask clarifying questions*, *fix mistakes*, or *use tools* in multiple steps. This requires multi-turn interaction within an episode/rollout.

Libraries like [verifiers](https://github.com/PrimeIntellect-ai/verifiers) handle this through environment inheritance, where each layer adds new capabilities:

| Layer | What it adds |
|-------|--------------|
| **Environment** | Base protocol: `reset()`, `step()`, reward |
| **↳ MultiTurnEnv** | Conversation history, turn limits, stopping conditions |
| **↳ ToolEnv** | Parses tool calls, executes them, returns results |
| **↳ StatefulToolEnv** | Persistent state across tool calls |
| **↳ SandboxEnv** | Isolated execution environments |
| **↳ CodeEnv** | Code execution with safety boundaries |

The pattern is elegant: `MultiTurnEnv` turns a stateless `step()` into a conversation loop. `ToolEnv` parses special tokens and executes tools. Higher layers add sandboxing and code execution. As a thought exercise, if we were to implement a simple ReAct agent based chatbot, we would have to inherit from `ToolEnv` and fill in the available tool list, tool execution logic and conversation stopping condition.


# The Reward Engineering Challenge

The reward function determines training dynamics more than any other design choice. The field has learned hard lessons about reward hacking, with frontier models now actively manipulating evaluation code when given the opportunity. 

TODO: cite SakanaAI & o3 monkey patching -> add citations.

## The Strict Reward Trap

Binary rewards worked surprisingly well for [Deepseek-R1](https://arxiv.org/abs/2501.12948), but in practice, strict binary rewards on complex tasks can often stall learning as there is no success signal to guide the learning. One mitigation would be to further finetune the LLM (SFT) on the task solutions before RL training. But beyond that, we usually need to shape the reward better.

## Partial Rewards: A Double-Edged Sword

<!-- TODO: mention this as PRM (Process Reward Modelling) -->
Giving **partial rewards for progress** seems like a good idea. In coding, perhaps give 0.1 reward for passing a single test case out of 10 (so 0.7 if it passes 7/10 tests). This dense feedback can help the model improve incrementally so that it's not all-or-nothing.

```python
# Seems reasonable...
reward = 0.1 * moved_forward + 0.5 * avoided_obstacle + 1.0 * finished
```

However, dense rewards come with a trap: **reward hacking**. The agent might find a way to exploit the reward structure rather than solving the actual task. This is particularly problematic in environments where looping or repetitive behavior can accumulate rewards.

### Reward Hacking 

A classic example comes from [OpenAI's research on faulty reward functions](https://openai.com/index/faulty-reward-functions/). In the boat racing game [CoastRunners](http://www.kongregate.com/games/longanimals/coast-runners), 

> The targets were laid out in such a way that the RL agent **could gain a high score without having to finish the course**. Instead of racing, it found an isolated lagoon where it could turn in a large circle and repeatedly knock over three targets, timing its movement to always hit them just as they respawned. Despite repeatedly catching fire, crashing into other boats, and going the wrong way on the track, the agent achieved a score **20% higher** than human players by completely ignoring the intended objective.

<figure style="max-width: 600px; margin: 0 auto;">
    <video controls style="width: 100%; height: auto; border-radius: 8px;">
        <source src="{{ site.url }}/{{ site.baseurl }}/assets/vids/rl_envs/CoastRunners_rl.mov" type="video/quicktime">
        Your browser does not support the video tag.
    </video>
    <figcaption><b>Figure:</b> <i>OpenAI's CoastRunners agent exploiting the reward function—scoring higher by circling in a lagoon than by actually racing. <a href="https://openai.com/index/faulty-reward-functions/">Source: OpenAI</a></i></figcaption>
</figure>


#### Gaming the benchmarks

Reward Hacking has become alarmingly sophisticated. [METR research (June 2025)](https://metr.org/blog/2025-06-05-recent-reward-hacking/) found o3 reward-hacking in 100% of runs on [certain tasks](https://github.com/METR/RE-Bench/tree/main/ai_rd_optimize_llm_foundry), with 30.4% hacking rate across [RE-Bench](https://github.com/METR/RE-Bench) overall. Documented exploits include: monkey-patching `torch.cuda.synchronize` to fake faster runtimes, tracing the Python call stack to steal the grader's ground_truth tensor, patching evaluation functions to return "succeeded: True", and overwriting PyTorch's `__eq__` operator to always return `True`. 

<!-- ```
Total: 1.6                    Total: ∞ (loops forever!)
Expected behavior:          Reward-hacked behavior:
                            
   START                       START
     │                           │
     ▼                           ▼
  ┌─────┐                     ┌─────┐
  │Move │ +0.1                │Move │ +0.1
  │Fwd  │                     │Fwd  │
  └──┬──┘                     └──┬──┘
     │                           │
     ▼                           ▼
  ┌─────┐                     ┌─────┐
  │Avoid│ +0.5                │Turn │ ◄─────┐
  │Obst │                     │Left │       │
  └──┬──┘                     └──┬──┘       │
     │                           │          │
     ▼                           ▼          │
  ┌─────┐                     ┌─────┐       │
  │Finish│ +1.0               │Move │ +0.1  │
  │ Race │                    │Fwd  │───────┘
  └─────┘                     └─────┘
                              
``` -->

## Curriculum Training


RL training is most effective when tasks are neither too easy nor too hard. Curriculum training solves this problem by dividing training into a few manually-defined phases of increasing difficulty ([Wen et al., 2025](https://arxiv.org/abs/2503.10460); [Luo et al., 2025](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2); [Song et al., 2025](https://arxiv.org/abs/2503.17287)), but these are coarse-grained and
lack adaptivity. **Adaptive curriculum learning** addresses these issues by matching problem difficulty to the model's evolving capabilities. 


[AdaRFT (Shi et al., 2025)](https://arxiv.org/abs/2504.05520) maintains a target difficulty level $T$ that evolves based on recent rewards. When average reward exceeds target ($\beta=0.5$) (also proposed by [DOTS (Yifan et al., 2025)](https://arxiv.org/abs/2506.05316v1)), difficulty increases; otherwise, it decreases. Their approach uses an external LLM (Qwen 2.5 MATH 7B) to estimate difficulty based on the success rate over 128 attempts. They observed a 2x reduction in training steps while improving accuracy. 

<iframe 
  src="{{ site.url }}/{{ site.baseurl }}/assets/visualizations/rl_envs/adarft.html" 
  style="width: 100%; height: 800px; border: none; border-radius: 16px; margin: 24px 0;"
  loading="lazy"
  title="AdaRFT Pipeline Interactive Visualization">
</iframe>

[INTELLECT-3 (Prime Intellect Team, 2025)](https://arxiv.org/abs/2512.16144) uses a similar approach to dynamically adjust the difficulty of the training tasks based on the model's performance. In their case, they used `Qwen/Qwen3-4B-Thinking-2507` for difficulty estimation in the initial phase.

<iframe 
  src="{{ site.url }}/{{ site.baseurl }}/assets/visualizations/rl_envs/intellect3_curriculum.html" 
  style="width: 100%; height: 1000px; border: none; border-radius: 16px; margin: 24px 0;"
  loading="lazy"
  title="INTELLECT-3 Curriculum Training Interactive Visualization">
</iframe>

<!-- Specifically, we can start with easier tasks and gradually increase difficulty.  -->

<!-- Fine-tuning on problems that are too easy or too hard leads to poor learning outcomes. Instead, the model should be trained on problems whose difficulty is close to the model's current capability. -->

<!-- Prime Intellect maintains difficulty levels in their benchmarks, primarily based on solvability by smaller models like Qwen-4B: -->


<!-- Notes: By optimizing a policy model with reward signals that reflect task success, RFT enables more targeted
learning than supervised finetuning (SFT) alone. However, despite its promise, RFT remains sample-
inefficient and computationally expensive.

Staged curricula divide training into a few manually-defined phases of increasing
difficulty (Wen et al., 2025; Luo et al., 2025; Song et al., 2025), but these are coarse-grained and
lack adaptivity. Other methods use online data filtering, repeatedly rolling out and pruning training
samples until the model’s average reward meets a target threshold (Bae et al., 2025; Yu et al., 2025).
While this approach helps prevent the model from stagnating on problems that are either too easy or
too difficult, it is not truly adaptive and incurs significant rollout overhead. -->

<!-- The intuition is simple: learning is most effective when tasks
are neither too easy nor too hard. ADARFT formalizes this by maintaining a target difficulty level,
which increases or decreases based on recent reward feedback. At each step, the model is trained on
examples closest to this target, promoting a steady progression through solvable yet challenging tasks.
The full algorithm is outlined in Algorithm 1 -->



## Adaptive Environments

Similar to curriculum training, but the environment itself evolves. Instead of manual difficulty levels, use an LLM to update rubrics and task parameters based on training progress.

Recent research has formalized these approaches:

- **AdaRFT** ([Shi et al., 2025](https://arxiv.org/abs/2504.05520)): Adaptive Reinforcement Finetuning dynamically adjusts training problem difficulty based on the model's recent reward signals. If the model is struggling, it sees easier problems; if it's succeeding, difficulty increases automatically.

- **AdaCuRL** ([Li et al., 2025](https://arxiv.org/abs/2511.09478)): Integrates coarse-to-fine difficulty estimation with adaptive curriculum scheduling. It also incorporates a data revisitation mechanism to mitigate catastrophic forgetting-the model periodically revisits easier problems to retain earlier capabilities.

- **CAPO** ([Yang et al., 2025](https://arxiv.org/abs/2512.02580)): Curriculum Advantage Policy Optimization bootstraps imitation learning with positive-only advantage samples, using curriculum mechanisms to improve generalization across complex reasoning tasks.


TODO: write about this
[Software agents can self-improve via self-play RL](https://x.com/YuxiangWei9/status/2003541373853524347)
og-paper-> [arxiv for self-play RL](https://arxiv.org/abs/2512.18552)

# Sandboxing: The Unsung Hero

## Why Sandboxing Matters

You might think: "I'll just run the model's code directly." Here's why that's a terrible idea for RL training:

1. **Segfaults and crashes**: One bad piece of generated code shouldn't kill a 20-day training run
2. **Infinite loops**: The model generates `while True: pass` and your training hangs
3. **Resource exhaustion**: Memory bombs, fork bombs, disk filling
4. **Security**: Arbitrary code execution on your training cluster is... not great


### Why Scaffolds Matter:

Reference for EXACT lines in this section: [Why Benchmarking is Hard](https://epoch.ai/gradient-updates/why-benchmarking-is-hard) -> 
> Scaffolds continue to have an outsized impact. As agentic evals, such as SWE-bench Verified or RLI, become more common, one component becomes increasingly important: The scaffold, i.e., the software that operates the agent, usually a CLI such as Claude Code, OpenHands, etc


> On SWE-bench Verified, a popular agentic coding benchmark, simply switching the scaffold makes up to an 11% difference for GPT-5 and up to a 15% difference for Kimi K2 Thinking. We cover the effect of the scaffold in our SWE-bench Verified review. The choice of scaffold has the single biggest impact on the overall performance

## The Scale Challenge

For efficient RL training, you need to run thousands of environment instances in parallel. Prime Intellect reports running 4,000 concurrent sandboxes during their RL training.

```python
# Naive approach: sequential execution (slow!)
for prompt in prompts:
    response = model.generate(prompt)
    reward = env.step(response)  # Each step might take seconds

# Parallel execution with sandboxing
async def run_parallel_envs(prompts, num_workers=4000):
    async with SandboxPool(num_workers) as pool:
        tasks = [pool.execute(env.step, p) for p in prompts]
        results = await asyncio.gather(*tasks)
    return results
```

## Beyond Python: Multi-Language Environments

The challenge multiplies when you consider:
- **Different programming languages**: Python, JavaScript, Rust, Go...
- **Database environments**: Testing SQL queries safely
- **CLI environments**: Shell commands, file system operations
- **SWE environments**: Full development setups with git, package managers, build tools
- **Computer use**: GUI interactions, browser automation

MCP (Model Context Protocol) servers can help for simple Python execution, but don't scale to these diverse requirements. This is why teams like Prime Intellect invest heavily in robust sandboxing infrastructure.

# Interesting Training Patterns

## Self-Generated Verification

An intriguing pattern (discussed with colleague Leonard):

**Pattern 1: LLM generates code with test cases**
```python
# Model generates both the solution and tests
generated_code = model.generate("Write a function to sort a list")
generated_tests = model.generate("Write test cases for this sort function")

# Cross-validate
reward = run_tests(generated_code, generated_tests)
```

**Pattern 2: LLM generates test cases with code as verification**
```python
# Flip it: generate tests first, then code
generated_tests = model.generate("Write edge case tests for sorting")
generated_code = model.generate("Write code that passes these tests")

# The most discriminative tests are those that fail most attempts
# This helps identify robust test cases
```

This bidirectional approach can help "robustify" both code generation and test generation capabilities.

Related research in this area:

- **CodeRL** ([Le et al., 2022](https://arxiv.org/abs/2207.01780)): Uses execution feedback from unit tests as reward signals, training a critic model to predict functional correctness and guide code generation.

- **Self-Training for Tool Use** ([Luo et al., 2024](https://arxiv.org/abs/2401.12999)): Shows that LLMs can learn to use tools without human demonstrations by generating their own training data through exploration-the model generates tool-use traces and learns from successful executions.

## Environment Groups

Not all tools are created equal. Colleague Max Meuer reported an interesting finding: **benchmark scores regress when too many different tools are introduced** to a Chat Environment. When there's only one tool, performance is higher.

This suggests organizing environments into capability groups:
- **Single-tool environments**: Master one tool at a time
- **Tool family environments**: Related tools (all file operations, all web APIs)
- **Full environments**: All tools available (but harder to train)

```python
class EnvironmentGroup:
    def __init__(self, name: str, tools: List[Tool]):
        self.name = name
        self.tools = tools
    
    @staticmethod
    def file_ops() -> 'EnvironmentGroup':
        return EnvironmentGroup("file_ops", [ReadFile, WriteFile, ListDir])
    
    @staticmethod  
    def web_apis() -> 'EnvironmentGroup':
        return EnvironmentGroup("web", [HttpGet, HttpPost, ParseJSON])
```

# Synthetic Tools and Data

The [Kimi K1.5 technical report](https://arxiv.org/abs/2501.12599) highlights an important insight: the diversity and quality of synthetic tool-use data matters enormously for tool proficiency. Moonshot AI invested heavily in generating massive amounts of synthetic tool interactions for training.

Key considerations:
- **Tool diversity**: Expose the model to many different tool interfaces (Kimi trained on thousands of unique tool signatures)
- **Error cases**: Include examples where tool calls fail or return unexpected results
- **Composition**: Multi-step tool use patterns where the output of one tool feeds into another
- **Edge cases**: Unusual parameter combinations, empty results, timeouts
- **Realistic distributions**: Tool usage patterns should mirror real-world applications

## LLMs as Tool Mocks

An interesting research direction: using LLMs to simulate tool behavior during training. This allows:
- Training without actual API access
- Generating diverse tool responses
- Simulating edge cases and errors

Research in this space:

- **ToolLLM** ([Qin et al., 2023](https://arxiv.org/abs/2307.16789)): Created a benchmark with 16,000+ real-world APIs across 49 categories. They used ChatGPT to generate diverse tool-use scenarios and demonstrated that training on synthetic API interactions transfers to real tool use.

- **APIGen** ([Liu et al., 2024](https://arxiv.org/abs/2406.18518)): An automated pipeline for generating verifiable, diverse function-calling datasets. Uses multi-stage verification (format checking, execution validation, semantic verification) to ensure quality of synthetic tool-use data.

## Tool Collection and MCP Servers

The ecosystem is growing with collections of tools and MCP (Model Context Protocol) servers that provide standardized interfaces for various capabilities.

Notable work in this area:

- **ToolBench** ([Qin et al., 2023](https://arxiv.org/abs/2307.16789)): Open-sourced a large-scale tool-use benchmark spanning 16,000+ APIs. This provides a standardized way to evaluate and train models on diverse tool interactions.

- **ToolACE** ([Liu et al., 2024](https://arxiv.org/abs/2401.06201)): Focuses on automated tool-use capability enhancement, demonstrating how to scale tool training data generation while maintaining quality through automated verification.

- **SpecTool** ([Kokane et al., 2024](https://arxiv.org/abs/2411.13547)): A benchmark for characterizing errors in tool-use LLMs. Identifies common failure patterns (parameter hallucination, format errors, semantic misunderstandings) and provides frameworks for systematic error mitigation.

The MCP (Model Context Protocol) ecosystem is also expanding rapidly, with standardized interfaces making it easier to create interoperable tool environments. However, the challenge remains: more tools often means worse performance per tool, suggesting that training strategies need to account for tool complexity and interactions.

# Putting It All Together

Here's a complete example of a multi-turn code environment:

```python
class MultiTurnCodeEnv(SandboxEnv):
    def __init__(
        self,
        problem: str,
        test_cases: List[TestCase],
        max_turns: int = 5,
        sandbox_config: SandboxConfig = None
    ):
        super().__init__(sandbox_config)
        self.problem = problem
        self.test_cases = test_cases
        self.max_turns = max_turns
        self.attempts = []
    
    def reset(self) -> str:
        self.attempts = []
        return f"""Problem: {self.problem}
        
Write a solution. You can iterate based on test feedback.
When done, wrap your final code in <final_code></final_code> tags."""
    
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        self.attempts.append(action)
        
        # Check for final submission
        if "<final_code>" in action:
            code = self.extract_final_code(action)
            results = self.sandbox.execute(code, self.test_cases)
            passed = sum(1 for r in results if r.passed)
            reward = passed / len(self.test_cases)
            return "", reward, True, {"results": results, "attempts": len(self.attempts)}
        
        # Run intermediate code and provide feedback
        code = self.extract_code(action)
        results = self.sandbox.execute(code, self.test_cases)
        
        feedback = self.format_feedback(results)
        
        if len(self.attempts) >= self.max_turns:
            passed = sum(1 for r in results if r.passed)
            reward = passed / len(self.test_cases)
            return "", reward, True, {"reason": "max_turns"}
        
        return feedback, 0.0, False, {}
    
    def format_feedback(self, results: List[TestResult]) -> str:
        lines = ["Test Results:"]
        for i, r in enumerate(results):
            status = "✓ PASS" if r.passed else "✗ FAIL"
            lines.append(f"  Test {i+1}: {status}")
            if not r.passed:
                lines.append(f"    Expected: {r.expected}")
                lines.append(f"    Got: {r.actual}")
                if r.error:
                    lines.append(f"    Error: {r.error}")
        return "\n".join(lines)
```

# Conclusion

Building effective RL environments for LLM training requires thinking beyond simple prompt-response pairs. Key takeaways:

1. **Pick the right verification**: Stdin/stdout, assertions, functional-choose based on your data and goals
2. **Design for failure**: Sandboxing isn't optional at scale
3. **Reward carefully**: Sparse rewards can halt training, dense rewards invite hacking
4. **Curriculum helps**: Gradual difficulty increases keep learning moving
5. **Group wisely**: Too many tools at once can hurt performance
6. **Scale matters**: Parallel environment execution is essential for efficient training

The environment is where your model learns. Invest in getting it right.

---

# References

**Curriculum & Adaptive Learning:**
- [AdaRFT: Efficient Reinforcement Finetuning via Adaptive Curriculum Learning](https://arxiv.org/abs/2504.05520) - Shi et al., 2025
- [AdaCuRL: Adaptive Curriculum Reinforcement Learning](https://arxiv.org/abs/2511.09478) - Li et al., 2025
- [CAPO: Curriculum Advantage Policy Optimization](https://arxiv.org/abs/2512.02580) - Yang et al., 2025

- [DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](https://arxiv.org/abs/2511.19399)
- [RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments](https://arxiv.org/abs/2511.07317)

**Code & Tool Use:**
- [CodeRL: Mastering Code Generation through Pretrained Models and Deep RL](https://arxiv.org/abs/2207.01780) - Le et al., 2022
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789) - Qin et al., 2023
- [APIGen: Automated Pipeline for Generating Verifiable Function Calling Datasets](https://arxiv.org/abs/2406.18518) - Liu et al., 2024
- [SpecTool: A Benchmark for Characterizing Errors in Tool-Use LLMs](https://arxiv.org/abs/2411.13547) - Kokane et al., 2024

**Self-Training & Synthetic Data:**
- [Self-Training Large Language Models for Tool Use](https://arxiv.org/abs/2401.12999)
- [How Kimi K2 Became One of the Best Tool-Using Models](https://www.dbreunig.com/2025/07/30/how-kimi-was-post-trained-for-tool-use.html)

If you have any questions, feel free to reach out on [Linkedin](https://www.linkedin.com/in/murali-manohar/), [Twitter](https://twitter.com/gitlostmurali) or [Mail](mailto:kmanoharmurali@gmail.com).


# Refernces

- [Intellect-3-report](https://arxiv.org/abs/2512.16144)
