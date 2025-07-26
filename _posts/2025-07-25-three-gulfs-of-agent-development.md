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

<figure>
  <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/regression_evals.jpg">  <img src="{{ site.url }}/{{ site.baseurl }}/assets/images/regression_evals.jpg" alt="Meme on regression"></a>
  <figcaption>
    <p>
      <strong>Figure 1:</strong> How adding new features without regression looks like. 
    </p>
  </figcaption>
</figure>

---

## Background & Motivation

LLM adoption is rocketing ahead of our ability to systematically track regressions. I once asked an engineer shipping an “agentic hot‑shot” product how they benchmarked it. The answer: “We use VIBES—Very Intelligent Business Evaluation Score.” A manual inspection like vibe-check may work in the initial phases but isn't suffice in the development cycle. There's a need for identifying the active players in the development cycle and bridging gaps between them.

## Agents' Non-Deterministic Behavior

Agents are an orchestration of LLMs, tools, memory, and business logic. As soon as you wire in a calculator tool, a single query like 24 + 28 can fork three ways:

1. The agent calls the tool and returns 52—nice.


2. It calls the tool with the wrong schema, then guesses the answer itself.


3. It decides doing math is “small potatoes,” skips the tool, and still answers.


Because this non-determinism makes failures hard to localize, it helps to frame agent development as a three-party juggle (data, developers, and LLMs) and to look for misalignments across them, the *Three Gulfs* ([Shankar et al. (2024)](https://arxiv.org/abs/2504.14764)).

---

## Gulf #1 — Data ↔ Developer (Gulf of Comprehension)

When devs stare at numbers on the evaluation dashboards, they’re really just peeking through a keyhole. Let's go through a few examples to understand why numbers alone can't provide a narrative of failure modes:

### 1. Next‑sentence suggestion system

* To evaluate the recommendation model's performance in real time, we can use a UX (User experience) metric like User Acceptance rate of the sentence suggestions. 

Metric-1; Acceptance Rate: % of sentence suggestions a user accepts.

* Another metric can be more nuanced like tracking the number of changes to the accepted suggestions. 

Metric-2; Edit Distance: Number of edits made by the User to an accepted suggestion. 

When the system is deployed and tracked, users often edited the suggestions. This would mean the model is bad. BUT, in reality, users who write shaky English often edit good suggestions into worse ones.

**Outcome:** Acceptance Rate tanks, devs panic—until manual review shows the model was fine, the users weren’t.


### 2. Cursor IDE memory prompts

Cursor launched a new feature where it tries to infer the Latent user preferences and store them as preferences. 

The model nails relevant recommendations but I keep clicking “Deny” because of privacy concerns or I don't want project specific preferences to be applied over all projects.

The metric signals failure; reality says otherwise.


**Takeaway:** Eval Dashboards cannot faithfully reflect the failure modes. Schedule routine manual error dives to ground‑truth what the data really means. Manual error analysis is a pill every Developer must consume. 


---

## Gulf #2 — Developer ↔ Agent (Gulf of Specification)

Humans are awful at giving precise instructions. Picture a recipe chatbot where the system prompt says: “Suggest easy recipes.” What does easy mean?

≤ 10 ingredients?

30‑minute cook time?

One‑pot only?

From recent performance regression checks ainst an agent, I observed the token usage to spike by 300% or 3x purely because the instructions were vague and contradictory.
Because you were vague, the agent rambles—burning more chain‑of‑thought tokens as it goes into self-monologue to de-clutter the contradictions in your prompt. 

Fixes:

1. Write spec tables: attribute · constraint · example.


2. Include both positive and negative exemplars right in the system prompt.


3. Track token usage per request; it’s a cheap regression alarm.


---

## Gulf #3 — Data ↔ Agent (Gulf of Generalization)

No matter how solid your system prompt is, clever users will jailbreak and coax the model into toxicity, policy leaks, or worse. Edge‑cases evolve faster than guardrails. This applies to any downstream tasks. We cannot generalize a model to handle 100% cases.

The Gulf of Generalization will never fully close, but you can narrow it. Monitor distribution drift—who is using your product and how. And iteratively fix your agent. 

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

