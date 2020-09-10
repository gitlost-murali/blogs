---
title: "Speeding-up your dataset-creation process"
excerpt: "Talks about building a dataset from scratch and offers a few tricks to speed-up the process"
tags: [Engineering, Tooling, Dataset creation, Annotation, Machine Learning, Deep Learning]
mathjax: true
categories: tooling annotation
---

## Introduction

Unlike most academia, Industrial problems won't come with a gold-standard dataset. Also, Dataset creation is not valued heavily in Industrial setting because it's perceived more as a `pain-in-the-xxx` than a `part-of-the-solution` for numerous reasons. Few of them are,

1. It is a manually taxing procedure.
2. You don't know how much data is enough. One must __push deadlines__ for the project since there's a clear dependency on the data.
3. Your Data Scientists might take it to their ego if asked to spend time on labeling data. (_Note: If you're one of them, spend time in atleast understanding the edge cases._)
 
> Telling that you need a lot of training data to get a high quality ML model is like saying __"You need a lot of money to be rich"__. - [Christopher Ré, Snorkel](https://www.youtube.com/watch?v=yu15Nf5eJEE)

While the data-dependency sounds like a redundant fact, it comes with an operational overhead. So, this post talks about how to deal with this situation and speed up your project. It also highlights that the role of Data Scientists is not just to create a DL-Architecture at the end but to participate in the whole project-cycle.

## Automating the process

### 1. __Seed Data:__
 When the project starts, you don't want to make your Data-Scientists sit idle until the data arrives. Instead, make it a _continuous loop_ and involve your Data-Scientists from Day-1. 
 
Getting the seed data enables you to fix the pipeline first _i.e_ Training data format, annotation tool, the ML/DL architecture setup and the output format.
 
There are two options/way to get the seed-data,

    1. Manual intervention
    2. Automation

__Option #1__ -> Manual intervention:

This option is straightforward in asking annotators to label small amout of instances but this is not a desired scenario because

1. It demands a lot of effort from annotators in creating a gold-standard dataset. And this small data may cover only a small distribution of the data.
2. Your Data-Scientistis are still not putting any effort to understand the domain & problem. But the plus-side is that they can start working on the project-pipeline.

__Option #2__ -> Getting noisy data:

In this option, a set of observations on the data are given by the domain-expert to Data-Scientists,  expecting them to __translate observations into rule-based functions__ for annotating the unlabelled instances. Although not perfect, this is solving two things for you,

1. Domain expertise is infused into the pipeline and this serves as your baseline.
2. Converting observations to code helps the data-scientist in understanding the problem better by glancing at the distribution of the data. Although I hate the case of __prognosticating accuracy__ before modelling, this approach helps a Data-Scientist to understand how easy/tough the problem is and to predict the accuracy to some certain extent.

So, considering the advantages above, we'll be focussing on developing __Option #2__ in this post.

__Note:__ If rule-based functions are magically covering the whole data distribution, we wouldn't be needing an [__AI Savior__](https://www.shreya-shankar.com/ai-saviorism/). And we must admit that this seed data is no where an exact approximation of the whole data-distribution. So, this is where we need a Human-in-loop interface to enhance the seed-data.

### 2. __Human-in-loop__: 

With the predictions from rule-based functions, you have two objectives now,

1. Re-annotating the noisy predictions for training an ML model.
2. Refining your rule-based functions for a robust baseline. This may even outperform your ML model.

#### 2.1 Tools for annotation:
These are exciting times to live. There are some great open-source annotation tools for us to use. Pick the one that suits your requirement. For this post, I'm taking `Named-Entity-Recognition` as the task and [`label-studio`](https://labelstud.io/) as the annotation tool.

{% include figure image_path="/assets/images/label-studio-val.png" alt="Label-Studio visual" caption="__Figure 1:__ NER prediction using [`Label-Studio`](https://labelstud.io/)." %}

#### 2.2 Pipeline for annotation
In addition to re-annotating your instances, validate your predictions into `Yes/No` category. If it's a YES, you can send them straight into an ML model. Otherwise, depending on your bandwidth, either update the rules again or just train the ML model on re-annotated/`NO`-labelled instances.

    YES-> Rules succeded at identifying entities.

    NO->  Rules failed at identifying entities.

{% include figure image_path="/assets/images/plan-for-data-creation.jpg" alt="Plan or pipeline" caption="__Figure 2:__ Pipeline from rule-based functions to ML model." %}

Make sure that your pipeline of _Seed-Data -> Annotation Tool -> Verified Data_ is connected properly so that you can do as many as iterations as possible.

##### 2.2.1 Visualizing the steps 

Your noisy data at the end of Step-2 would look something like this,
{% include figure image_path="/assets/images/label-studiostep2-pipeline.png" alt="Noisy Data" caption="__Figure 3:__ Noisy Data Format" %}

After validating & re-annotating your noisy data, this is the output you'd be getting for Step-3 (left) & Step-4 (right),
<figure class="half">
	<img src="{{ site.url }}/{{ site.baseurl }}/assets/images/label-studiostep2-yes-pipeline.png">
	<img src="{{ site.url }}/{{ site.baseurl }}/assets/images/label-studiostep2-no-pipeline.png">
	<figcaption><b>Figure 4:</b> <i>Left -></i> Instances that are correctly predicted by rules. <i>Right -> </i> Failed to detect by Rules</figcaption>
</figure>

__Note:__ The objective of using `Yes/No` is update the rule-based functions. We re-annotate the wronly-predicted instances but validate it as `No` to let the Data-Scientist know where rules fail.

## Conclusion

__Q)__ _Demanding domain-expertise for rule-based functions sounds like we're dating back to the pre-deep-learning era of feature-engineering. Isn't it counter-productive?_

__Ans.__ Our objective of rules is to support the ML model not deliver a complete rule-based solution. For instance, let's consider an example,

> __Uber__ raised $1 million from the initial IPO.

From a rules perspective, you might just do a lookup on 100-popular companies and tag any sub-string that matches the company as an entity. But from the ML perspective, it considers the context rather than just the word. Hence, if 60% of your data can be covered with rules and we train an ML model on the 60%, chances are that ML will bring in a deeper-perspective to cover the remaining 40%.

__Q)__ _Any alternatives to rule-based functions?_

__Ans.__ If an off-shelf ML model matches the problem or its domain, one can use that for the 1st iteration of data.

That's it from my side. Hope you find this post useful.

Thanks,

_Murali Manohar_.