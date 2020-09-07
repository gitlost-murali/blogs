---
title: "Expediting your dataset-creation process"
excerpt: "Talks about building a dataset from scratch. And the tricks to consider."
tags: [Engineering, Tooling, Dataset creation, Annotation, Machine Learning, Deep Learning]
mathjax: true
categories: tooling annotation
---

## Introduction

Unlike most academia situations, Industrial problems don't come with a gold-standard dataset. Also, Dataset creation is not valued heavily in Industrial setting because it's perceived more as a `pain-in-the-xxx` than a `part-of-the-solution` for numerous reasons. Few of them are,

1. It is a manually taxing procedure. One should push deadlines for the project since there's a clear dependency on the data.
2. You don't know how much data is enough.
3. Your Data Scientists might take it to their ego if asked to spend time on labeling data. (_Note: If you're a data scientist, don't do it. Spend your time atleast understanding the edge cases._)
 
> Telling that you need a lot of training data to get a high quality ML model is like saying __"You need a lot of money to be rich"__. - [Christopher Ré, Snorkel](https://www.youtube.com/watch?v=yu15Nf5eJEE)

While it sounds like a redundant fact, it comes with an operational overhead. So, this post talks about how to deal with this situation and speed up your project. It also highlights that the role of Data Scientists is not just to create a DL-Architecture at the end but to participate in the whole project-cycle.

## Automating the process

1. __Seed Data:__ When the project starts, you don't want to make your Data-Scientists sit idle until the data is arrived. Instead, make it a _continuous loop_. You have two options here,

    1. Manual intervention
    2. Automation

Getting seed data enables you to fix the pipeline first _i.e_ Training data format, annotation tool, the ML/DL architecture setup and the output format.

Option #1 -> Manual intervention:

This option is straightforward in asking annotators to label small amout of instances but this is not a desired scenario because
    1. It demands a lot of attention from annotators in creating a gold-standard dataset.
    2. Your Data-Scientistis are still not putting any effort to understand the domain & problem.

Option #2 -> Getting noisy data:

In this option, your domain-expert will give the DScientist a set of observations on the data expecting DScientists to __translate them into rule-based functions__ for annotating the unlabelled instances. Although not perfect, this is solving two things for you,
    1. Domain expertise is infused into the pipeline.
    2. Converting observations to code helps the data-scientist in understanding the problem better by glancing the distribution of the data. Although I hate the case of prognosticating accuracy by sales/management team, this approach helps you as a Data-Scientist to understand how easy/tough the problem is and predict the accuracy to some certain extent.

Considering the advantages above, let's focus only about __Option #2__ in this post.

If rule-based functions are doing the magic and covering the whole data distribution, we wouldn't be needing an __AI Savior__. We must admit that this seed data is no where an exact approximation of the whole data-distribution. So, this is where we need a Human-in-loop interface to enhance the seed-data.

2. __Human-in-loop__: 

Send 

These are exciting times to live. There are some great open-source annotation tools for us to use. Pick the one that suits your requirement. Make sure that your pipeline of _Seed-Data -> Annotation Tool -> Verified Data_ is strong enough.



## Appendix

Demanding domain-expertise might sound like we're dating back to the pre-deep-learning era of feature-engineering.

That's it from my side. Hope you find this post useful.

Thanks,

_Murali Manohar_.