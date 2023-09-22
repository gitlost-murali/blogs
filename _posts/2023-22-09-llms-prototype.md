---
title: Rapid Prototyping with Large Language Models - A New Era of Product Integration
excerpt: Exploring the potential of GPT and LLMs in product development and testing innovative solutions.
tags: [Machine Learning, Language Models, GPT, LLM, Product Development, Prototyping, huggingface, transformers]
date: 2023-09-22 05:28:10 +0530
categories: machine-learning product-development
toc: true
permalink: /:categories/:title
---

In the ever-evolving landscape of technology, the ability to rapidly prototype and test ideas is invaluable. With the advent of GPT and other Large Language Models (LLMs), businesses now have a powerful tool at their disposal to accelerate this process. In this blog, I'll share some of the innovative ways we've integrated GPT and LLMs into our products at AskUI, demonstrating their potential in real-world applications.

## A Bit About AskUI, who's the target company

AskUI is more than just a workplace for me; it's where innovation meets automation. While many might think of Selenium when they hear "user automation," AskUI goes beyond. Traditional tools like Selenium are dependent on the underlying website's code. But what happens when the code changes? Or when you need to automate tasks on desktop applications? Enter AskUI. We leverage vision-based techniques, akin to human perception, to identify elements. Through object detection and other advanced methods, we've created a robust automation tool that's not just limited to web applications.

Natural Language to Actions

One of the standout features of AskUI is its intuitive Domain Specific Language (DSL). For instance, a command like aui.click().button().withText("Nett Hier aber ...").exec(); is self-explanatory. However, as we aim to cater to a broader audience, including analysts at EY or Goldman Sachs, we realized the need for a more natural interaction. The goal? Convert natural language commands into AskUI DSL.

Instead of diving straight into building a machine translation model, we turned to GPT and LLMs. By feeding them our documentation and list of commands, we were able to quickly prototype a system that translates natural language into our DSL. The results were beyond translation; GPT demonstrated its capability to plan and generate entire workflows based on an end goal.

Streamlining Workflows with Vision

Building on this idea, we envisioned a system where users wouldn't even need to type commands. With AskUI's workflows, users can define a series of steps using screenshots. For each step, they specify the action, and our engine executes it. But what if we could automate this specification process?

By integrating GPT with our engine, we can automatically detect and map user clicks to specific elements on a screenshot. Using the company's documentation as context, GPT can then generate the corresponding DSL command. This approach not only streamlines the workflow creation process but also showcases GPT's ability to understand positional information and generate context-aware commands.

Domain-Specific Finetuning with LLMs

While GPT is powerful, there are instances where the input limit can be a constraint, especially when dealing with extensive documentation. This is where models like LLAMA-2 come into play. By finetuning these models on our domain-specific data, we can create a chatbot that's well-versed in our domain.

To generate a synthetic chat dataset, we leveraged GPT to create question-answer pairs based on our documentation. After finetuning, we also integrated a retrieval augmentation system. This system fetches relevant documents for each user query, providing the chatbot with the exact context it needs to generate a response.

Conclusion

The integration of GPT and LLMs into our product development process at AskUI has been transformative. These models have not only accelerated our prototyping process but have also opened up new avenues for innovation. Whether it's translating natural language to DSL, automating workflow creation, or building domain-specific chatbots, the possibilities are endless. As we continue to explore and experiment, one thing is clear: LLMs are set to play a pivotal role in the future of product development.