---
title: Building RL Environments for LLM Training - From Car Racing to Code Agents
excerpt: A deep dive into designing environments for reinforcement learning with LLMs - understanding actions, observations, rewards, and scaling challenges
tags: [Machine Learning, Language Models, Reinforcement Learning, Environments, Agents, GRPO, Training]
date: 2025-12-20 10:30:00 +0530
categories: machine-learning data-science
toc: true
permalink: /:categories/:title
---

# Background

Reinforcement learning follows the FAFO principle-Fool Around and Find Out ([more on this here]({{ site.url }}/{{ site.baseurl }}/grpo-intro/)). But to fool around, LLMs need a playground: an *environment* where they can take actions, observe outcomes, and learn from their mistakes. If you've ever played a video game, you already grasp the core idea. Think of a car racing game: your keyboard inputs (up/down/left/right) feed into the game engine, which executes a step and returns an outcome - did you crash, cross the finish line, or are you still racing? This action → outcome loop is exactly what RL environments provide for LLM training.

<figure style="max-width: 400px; margin: 0 auto;">
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/environments/car_arrows.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/environments/car_arrows.png" style="width: 100%; height: auto;"></a>
    <figcaption><b>Figure 1:</b> <i>A car racing game illustrating the RL loop: actions (arrows for up/left/right) lead to observations (track state) and rewards (progress towards finish line)</i></figcaption>
</figure>

This post is a deep dive into the world of RL environments for LLMs-how code gets verified, what makes multi-turn interactions tricky, and the infrastructure challenges that emerge at scale.

# The Environment Abstraction

## Actions and Observations

Every RL environment follows a simple contract:

```python
class Environment:
    def reset(self) -> Observation:
        """Reset the environment to initial state"""
        pass
    
    def step(self, action: Action) -> Tuple[Observation, Reward, Done, Info]:
        """Take an action and return the outcome"""
        pass
```

In our car racing analogy:
- **Action**: Your keyboard input (up/down/left/right)
- **Observation**: The current game state (car position, track layout, other cars)
- **Reward**: Points for moving forward, penalty for hitting walls
- **Done**: Whether the race is over (finished, crashed, or timed out)
- **Info**: Additional metadata (lap time, fuel remaining)

For LLM environments:
- **Action**: The model's generated text response
- **Observation**: The prompt, conversation history, tool outputs
- **Reward**: Score from verifier (correctness, helpfulness, etc.)
- **Done**: Whether to stop or continue interacting with the environment (task complete, max turns reached, etc.)
- **Info**: Execution traces, intermediate states

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              RL ENVIRONMENT                                │
├────────────────────────────┬───────────────────────────────────────────────┤
│         CAR RACING         │               LLM TRAINING                    │
├────────────────────────────┼───────────────────────────────────────────────┤
│ Action: ↑ ↓ ← →            │ Action: Generated text/code/toolcalls         │
│ Observation: Track state   │ Observation: Prompt + history + tool outputs  │
│ Reward: +1 forward         │ Reward: Verifier score                        │
│ Done: Finish/Crash         │ Done: Task complete / max turns               │
│ Info: Lap time             │ Info: Execution trace + tool outputs          │
└────────────────────────────┴───────────────────────────────────────────────┘
```

# Verification: The Heart of RL Environments

Every environment needs to answer one question: *"Did the model do it right?"*

For math, it's straightforward-extract the answer, compare to ground truth. The verification can be rule-based (string matching, numerical comparison) or LLM-based for complex reasoning tasks.

Code verification is more nuanced. Here are the main approaches:

**Stdin/Stdout Matching**

Run the code with specific inputs, compare stdout to expected output.

```python
def verify_stdin_stdout(code: str, test_case: TestCase) -> bool:
    result = sandbox.run(code, stdin=test_case.input)
    return result.stdout.strip() == test_case.expected_output.strip()
```

**Assertion-Based Testing**

Wrap the solution in a test harness with assertions. If assertions pass, the code is correct.

```python
def verify_with_assertions(solution_code: str, test_code: str) -> bool:
    # test_code contains: assert solution([1,2,3]) == 6
    full_code = f"{solution_code}\n\n{test_code}"
    result = sandbox.run(full_code)
    return result.exit_code == 0  # No AssertionError = pass
```

**Functional Verification**

Call the generated function directly and check outputs match expected values. This is what you'd typically see in coding benchmarks-test cases specify function name, inputs, and expected output:

```json
{
  "func_name": "canSplitArray",
  "input": "[2, 2, 1]\n4",
  "output": "true"
}
```

```python
def verify_functional(code: str, test_cases: List[TestCase]) -> float:
    passed = 0
    for tc in test_cases:
        actual = sandbox.call_function(code, tc.func_name, tc.input)
        if actual == tc.output:
            passed += 1
    return passed / len(test_cases)
```

**Execution-Only Verification**

The most lenient-just check if the code executes without errors. Useful for creative coding tasks or as partial credit.

```python
def verify_runs(code: str) -> float:
    result = sandbox.run(code, timeout=5)
    return 1.0 if result.exit_code == 0 else 0.0
```

A typical code environment uses one of these paradigms:

```python
class CodeEnv(Environment):
    def __init__(self, problem: str, test_cases: List[TestCase]):
        self.problem = problem
        self.test_cases = test_cases
    
    def step(self, llm_code: str) -> Tuple[str, float, bool, dict]:
        results = []
        for tc in self.test_cases:
            # Could use any verification paradigm here
            passed = self.sandbox.run_with_assertions(llm_code, tc.assertions)
            results.append(passed)
        
        reward = sum(results) / len(results)
        return "", reward, True, {"passed": sum(results), "total": len(results)}
```

# When One Turn Isn't Enough

Simple verification works for isolated problems, but real-world tasks rarely complete in a single turn. The model might need to:
- Iterate on code based on test feedback
- Use tools to gather information
- Break down complex problems step by step

This is where multi-turn environments come in. Here's the hierarchy from Prime Intellect:

```
┌─────────────────┐
│   Environment   │  (base class)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  MultiTurnEnv   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│     ToolEnv     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ StatefulToolEnv │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   SandboxEnv    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    CodeEnv      │
└─────────────────┘
```

## MultiTurnEnv

The base for any conversational or agentic task:

```python
class MultiTurnEnv(Environment):
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turn_count = 0
        self.history = []
    
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        self.turn_count += 1
        self.history.append({"role": "assistant", "content": action})
        
        # Check stopping conditions
        if self.is_final_answer(action):
            return self.evaluate_final_answer(action)
        
        if self.turn_count >= self.max_turns:
            return "", 0.0, True, {"reason": "max_turns"}
        
        # Generate next observation
        observation = self.get_next_observation(action)
        self.history.append({"role": "user", "content": observation})
        
        return observation, 0.0, False, {}
```

## ToolEnv

Extends MultiTurnEnv with tool calling capabilities:

```python
class ToolEnv(MultiTurnEnv):
    def __init__(self, tools: List[Tool], **kwargs):
        super().__init__(**kwargs)
        self.tools = {t.name: t for t in tools}
    
    def get_next_observation(self, action: str) -> str:
        # Parse tool calls from action
        tool_calls = self.parse_tool_calls(action)
        
        results = []
        for call in tool_calls:
            tool = self.tools.get(call.name)
            if tool:
                result = tool.execute(call.arguments)
                results.append(f"{call.name}: {result}")
        
        return "\n".join(results)
```

## React Agent Environment

A classic example: the ReAct pattern with a stopping condition on "Final Answer":

```python
class ReactEnv(ToolEnv):
    def is_final_answer(self, action: str) -> bool:
        return "Final Answer:" in action or "</final_answer>" in action
    
    def evaluate_final_answer(self, action: str) -> Tuple[str, float, bool, dict]:
        extracted = self.extract_final_answer(action)
        reward = self.judge(extracted, self.expected_output)
        return "", reward, True, {"answer": extracted}
```

# The Reward Engineering Challenge

## The Strict Reward Trap

From experience: strict binary rewards (1 for correct, 0 for wrong) on difficult tasks like math can halt training entirely. The model never gets positive signal to learn from.

```python
# This can kill training on hard tasks
reward = 1.0 if perfectly_correct else 0.0
```

## Partial Rewards: A Double-Edged Sword

Partial rewards seem like the solution, but they introduce reward hacking:

```python
# Seems reasonable...
reward = 0.1 * moved_forward + 0.5 * avoided_obstacle + 1.0 * finished

# But the car learns to drive in circles, 
# accumulating infinite partial rewards!
```

The model exploits the reward structure rather than solving the actual task. This is particularly problematic in environments where looping or repetitive behavior can accumulate rewards.

```
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
                              
Total: 1.6                    Total: ∞ (loops forever!)
```

## Curriculum Training

One effective solution: start with easier tasks and gradually increase difficulty. Prime Intellect maintains difficulty levels in their benchmarks, primarily based on solvability by smaller models like Qwen-4B:

```python
class CurriculumScheduler:
    def __init__(self, tasks_by_difficulty: Dict[int, List[Task]]):
        self.tasks = tasks_by_difficulty
        self.current_level = 1
    
    def get_batch(self, success_rate: float) -> List[Task]:
        # Move to harder tasks when success rate is high
        if success_rate > 0.8 and self.current_level < max(self.tasks.keys()):
            self.current_level += 1
        elif success_rate < 0.2 and self.current_level > 1:
            self.current_level -= 1
        
        return random.sample(self.tasks[self.current_level], k=batch_size)
```

## Adaptive Environments

Similar to curriculum training, but the environment itself evolves. Instead of manual difficulty levels, use an LLM to update rubrics and task parameters based on training progress.

Recent research has formalized these approaches:

- **AdaRFT** ([Shi et al., 2025](https://arxiv.org/abs/2504.05520)): Adaptive Reinforcement Finetuning dynamically adjusts training problem difficulty based on the model's recent reward signals. If the model is struggling, it sees easier problems; if it's succeeding, difficulty increases automatically.

- **AdaCuRL** ([Li et al., 2025](https://arxiv.org/abs/2511.09478)): Integrates coarse-to-fine difficulty estimation with adaptive curriculum scheduling. It also incorporates a data revisitation mechanism to mitigate catastrophic forgetting-the model periodically revisits easier problems to retain earlier capabilities.

- **CAPO** ([Yang et al., 2025](https://arxiv.org/abs/2512.02580)): Curriculum Advantage Policy Optimization bootstraps imitation learning with positive-only advantage samples, using curriculum mechanisms to improve generalization across complex reasoning tasks.

# Sandboxing: The Unsung Hero

## Why Sandboxing Matters

You might think: "I'll just run the model's code directly." Here's why that's a terrible idea for RL training:

1. **Segfaults and crashes**: One bad piece of generated code shouldn't kill a 20-day training run
2. **Infinite loops**: The model generates `while True: pass` and your training hangs
3. **Resource exhaustion**: Memory bombs, fork bombs, disk filling
4. **Security**: Arbitrary code execution on your training cluster is... not great

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

**Code & Tool Use:**
- [CodeRL: Mastering Code Generation through Pretrained Models and Deep RL](https://arxiv.org/abs/2207.01780) - Le et al., 2022
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789) - Qin et al., 2023
- [APIGen: Automated Pipeline for Generating Verifiable Function Calling Datasets](https://arxiv.org/abs/2406.18518) - Liu et al., 2024
- [SpecTool: A Benchmark for Characterizing Errors in Tool-Use LLMs](https://arxiv.org/abs/2411.13547) - Kokane et al., 2024

**Self-Training & Synthetic Data:**
- [Self-Training Large Language Models for Tool Use](https://arxiv.org/abs/2401.12999) - Luo et al., 2024
- [Kimi K1.5 Technical Report](https://arxiv.org/abs/2501.12599) - Moonshot AI, 2025

---

If you have any questions, feel free to reach out on [Linkedin](https://www.linkedin.com/in/murali-manohar/), [Twitter](https://twitter.com/gitlostmurali) or [Mail](mailto:kmanoharmurali@gmail.com).

