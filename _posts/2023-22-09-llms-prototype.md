---
title: Rapid Prototyping with LLMs - The Journey from Notebook Musings to Real-World Enterprise Solutions
excerpt: Talks about how LLMs enabled me to rapidly prototype and test ideas and how they are enabling a new era of Product Integration. 
tags: [Machine Learning, Language Models, GPT, LLM, Product Development, Prototyping, huggingface, transformers]
date: 2023-09-22 05:28:10 +0530
categories: machine-learning product-development
toc: true
permalink: /:categories/:title
---

# Introduction

The ability to rapidly prototype and test ideas is invaluable, especially in the ever-evolving technology landscape. With the advent of GPT and other Large Language Models (LLMs), businesses now have a powerful tool at their disposal to accelerate this process. In this blog, I'll share some of the ways I've integrated LLMs into enterprise products, demonstrating their potential in real-world applications.

## Target Enterprise

[__AskUI__](https://www.askui.com/) is a UI automation tool. While many might think of Selenium when they hear "user automation," AskUI goes beyond it. Traditional tools like Selenium are dependent on the underlying website's code. But what happens when the code changes? Or when you need to automate tasks on desktop applications? Enter AskUI, which leverage vision-based techniques, similar to human perception, to identify elements. So, it sees things like a human and you can ask it like you ask your testing team to click on a red button or signup button. Through object detection and other advanced methods, the team has created a robust automation tool that's not just limited to web applications.

## Natural Language to Actions

One of the standout features of AskUI is its __intuitive__ Domain Specific Language (DSL). For instance, a command like `aui.click().button().withText("Hello World").exec();` is self-explanatory. However, as we aim to cater to a broader audience, including analysts at Goldman Sachs or a common man, we realized the need for a more natural interaction. The goal? Convert natural language commands into AskUI DSL.

Instead of diving straight into building a machine translation model, we turned to GPT and LLMs. By feeding them our documentation and list of commands, we were able to quickly prototype a system that translates natural language step into our DSL. We just had to provide the existing functions (like get, await, etc.), end goal, and GPT would generate the entire workflow. For instance, if the goal was to "click on the red button," GPT would generate the following DSL commands: `aui.click().button().withText("red").exec();`.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/blog-llm-prototype/nli2dsl.png/"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/blog-llm-prototype/nli2dsl.png/"></a>
    <figcaption><b>Figure 1:</b> Natural Language to DSL </figcaption>
</figure>

The results were beyond translation; GPT demonstrated its capability to plan and generate entire workflows based on an end goal. This was a game-changer for us. We could now leverage GPT to generate workflows for our users, who could then tweak them as needed. This approach not only saves time but also enables users to focus on the end goal rather than the steps required to achieve it.

## Streamlining Workflows with Vision

Building on this idea, we envisioned a system where users wouldn't even need to type commands. With AskUI's workflows, users can define a series of steps using screenshots. For each step, they specify the action, and our engine executes it. But what if we could automate this specification process?


Specifically, AskUI inference engine detects all UI elements that are visible on the screen. By integrating AskUI's inference engine with GPT, we can automatically detect all elements on a screenshot and map the user clicks to specific elements. Using the relevant product documentation as context, GPT can then generate the corresponding DSL command. This approach not only streamlines the workflow creation process but also showcases GPT's ability to understand positional information and generate context-aware commands.

<figure style="max-width: 100%; width: 100%;">
<video controls style="max-width: 100%; width: 100%; height: auto;">
    <source src="{{ site.url }}/{{ site.baseurl }}/assets/vids/llm_prototype/easyworkflow.mp4/" type="video/mp4">
</video>
    <figcaption><b>Figure 2:</b> Easy Workflow Creation </figcaption>
</figure>

## Domain-Specific Finetuning with LLMs

Having a dedicated DSL requires users to be aware of all the functions. Even if a product releases extensive documentation, it is often the case that users don't go through all of it. One way to assist them would be to have a chatbot/search engine which suggests what functions/tricks are available and can be used.

While GPT is capable of being a potential chatbot, there are instances where the input text limit can be a constraint, especially when dealing with extensive documentation. It is not possible to fit the entire documentation into the input context limit. This is where the open-source LLMs like LLAMA-2 come into play. By finetuning these models on our domain-specific data, we can create a chatbot that's well-versed in our domain.

To generate a synthetic chat dataset, we leveraged GPT to create question-answer (QA) pairs based on our documentation. Specifically, we parse through each document and ask GPT to generate possible QA pairs, simulating conversations of users asking questions about debugging, tools, etc. In addition to just passing the current document, we also pass the last _k_ QA pairs generated for more context to GPT for improved QA pair generation. After creating the synthetic chat dataset, we trained LLAMA2-7B-chat model on it using [LoRA](https://gitlostmurali.com/machine-learning/data-science/lora-qlora) finetuning approach.


<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/blog-llm-prototype/chat1.png/"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/blog-llm-prototype/chat1.png/"></a>
    <figcaption><b>Figure 3:</b> Chat Interface of LLAMA2 chat finetuned on AskUI docs</figcaption>
</figure>

After finetuning the LLAMA2 model, we also integrated a retrieval augmentation system. This system fetches relevant documents for each user query, providing the chatbot with the exact context it needs to generate a response. For the debate on RAG vs finetuning, here is a relevant [post by LlamaIndex](https://www.linkedin.com/posts/llamaindex_we-added-a-lot-of-new-fine-tuning-features-activity-7116229026754543616-i-Ab?utm_source=share&utm_medium=member_ios). For more details on RAG vs long context windows, you can checkout the recent paper, [RETRIEVAL MEETS LONG CONTEXT LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.03025.pdf).

## Conclusion

The integration of GPT and LLMs into our product development process at AskUI has been transformative. These models have not only accelerated our prototyping process but have also opened up new avenues for innovation. Whether it's translating natural language to DSL, automating workflow creation, or building domain-specific chatbots, the possibilities are endless. As we continue to explore and experiment, one thing is clear: LLMs are set to play a pivotal role in the future of product development.