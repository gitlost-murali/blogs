
---
title: "Bridging the Three Gulfs of Agentic Evaluation"
excerpt: "A practical framework for spotting and fixing evaluation blind spots in agentic LLM pipelines, based on Shankar et al.’s Three Gulfs model."
date: 2025-07-25
categories: Blog
Tags:
- LLM
- Agents
- Evaluation
- Metrics
toc: true
---

Bridging the Three Gulfs of Agentic Evaluation

> Agentic systems are the new hot‑shot in town—but evaluating them without losing your mind is still an unsolved puzzle.




---

TL;DR

Building agents means juggling data, developers, and the agents themselves. Shankar et al. (2024) call the mismatches between those three players the Three Gulfs. If you can spot and bridge each gulf, your eval game levels up fast.


---

## Background & Motivation

LLM adoption is rocketing ahead of our ability to systematically track regressions. I once asked an engineer shipping an “agentic hot‑shot” product how they benchmarked. The answer: “We use VIBES—Very Intelligent Business Evaluation Score.” Cool acronym, zero rigor.

Agents aren’t just fancy prompts; they’re an orchestration of LLMs, tools, memory, and business logic. As soon as you wire in a calculator tool, a single query like 24 + 28 can fork three ways:

1. The agent calls the tool and returns 52—nice.


2. It calls the tool with the wrong schema, then guesses the answer itself.


3. It decides doing math is “small potatoes,” skips the tool, and still answers.



That non‑determinism makes traditional integration tests crumble. Enter the Three Gulfs model—a map of where evaluation pain creeps in.


---

## Gulf #1 — Data ↔ Developer (Gulf of Comprehension)

When devs stare at dashboards, they’re really peeking through a keyhole. Example:

Grammarly‑style next‑sentence suggestion

UX metric · Acceptance Rate: % of suggestions a user accepts.

Observation: Users who write shaky English often edit good suggestions into worse ones.

Outcome: Acceptance Rate tanks, devs panic—until manual review shows the model was fine, the users weren’t.


Cursor IDE memory prompts

The model nails relevant recommendations.

I keep smashing “Deny” because of privacy nerves.

The metric signals failure; reality says otherwise.


Takeaway: Dashboards lie. Schedule routine manual error dives to ground‑truth what the data really means.


---

## Gulf #2 — Developer ↔ Agent (Gulf of Specification)

Humans are awful at giving precise instructions. Picture a recipe chatbot where the system prompt says: “Suggest easy recipes.” What does easy mean?

≤ 10 ingredients?

30‑minute cook time?

One‑pot only?


Because you were vague, the agent rambles—burning 3× more chain‑of‑thought tokens as it wrestles with contradictions in your prompt.

Fixes:

1. Write spec tables: attribute · constraint · example.


2. Include both positive and negative exemplars right in the system prompt.


3. Track token usage per request; it’s a cheap regression alarm.




---

## Gulf #3 — Data ↔ Agent (Gulf of Generalization)

No matter how solid your alignment stack, clever users will jailbreak and coax the model into toxicity, policy leaks, or worse. Edge‑cases evolve faster than guardrails.

**Remedies:**

Keep an adversarial eval suite that grows with every jailbreak you spot.

Rotate prompts + policies; static shields get reverse‑engineered.

Monitor distribution drift—who is using your product and how.


The gulf will never fully close, but you can narrow it.


---

## A Practical Workflow

```flowchart TD
    A[Analyze] --> B[Measure]
    B --> C[Improve]
```

1. Analyze: Run the agent on a sample set; label failures. Tag each bug to a gulf.


2. Measure: Turn those qualitative tags into numbers—precision, token cost, jailbreak rate, whatever moves the biz.


3. Improve: Patch prompts, tweak tools, swap models; fine‑tune only when cheaper fixes flop.



Rinse, repeat.


## Closing Thoughts & What’s Next

The Three Gulfs frame doesn’t magically grade your agent, but it tells you where to look when things go sideways. In the next post we’ll dig into prompt‑engineering recipes and eval metrics you can steal.

Got your own war stories? Drop ’em—let’s compare scars.

