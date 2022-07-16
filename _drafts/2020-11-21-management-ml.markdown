---
title: Revisiting the factors contributing to AI projects' failure
excerpt: Introspecting current state of ML projects. Discussing why and where they fail.
tags: [Industry, Deep Learning, Machine Learning, Management, Team Management]
date: 2020-11-20 21:30:10 +0530
categories: ml industry
permalink: /:categories/:title
---

["85% AI projects doesn't make it to production"](https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence#:~:text=Conversations%20with%20Gartner%20clients%20reveal,well%2Dprepared%20for%20implementing%20AI.&text=Gartner%20predicts%20that%20through%202022,teams%20responsible%20for%20managing%20them) has been an omnipresent figure that's churned extensively by the evangelists and VCs, ironically, to promote their AI solutions.

In this micro-blog/post, we revisit the reasons that pave way for failure:

1. __ML is still research__ -> It is too optimistic & maybe naive to achieve 100% success rate.
    1. While the field witnessed leaders progressing from _"80-85% percent accuracy tho ho jaayega"_ to _"We informed our clients about the limitations"_, being educated about AI limitations is just one part of the game. Scientific temparement is needed at multiple levels atleast until we can break open the "black-box" and move from "Sorcery" to "Science".
2. __Poor Project Management__ -> ML brings unique challenges in managing ML teams, thereby, making it a chaotic & confusing experience.
    1. __Orchestrating different teams__ i.e. Data Engineers, Data Scientists, Annotators & DevOps engineers is a herculian task. Some of these teams cannot be billed from Day-1 and need not be actively involved everyday. Ideally, Communication b/w these teams is expected to be linear, just like your data-flow. But its actually non-linear with feedback-loops and things get real messy, real soon.
    2. __Progress/Accuracy improvements__ are not linear in ML unless otherwise carefully planned. Questions like _"How much time do you need to push 70% accuracy to 90% accuracy?"_ are vague and simply not the right questions to ask. Instead, a well-informed leader would ask, _"What methods should we try and how much time each of it takes? What resources do you need to make it achievable?"_.
3. __Doomed to Fail__ -> Technically infeasible or poorly scoped. Ex: 
    1. Unsure about getting access to a bigger and private dataset. 
    2. Predicting if a person is an offender from his/her face.
    3. Having a false impression that identifying animal instances is of the same difficulty level as identifying animal species.

4. __Can't make the leap into production__ -> Unclear success criteria is another factor that lurks in the darkness until the project is ready for production. For example,
    1. __Interpretability__: After spending $$$ on AI projects' resources, few teams fall back to traditional algorithms, trading accuracy with explainability & interpretability.
    2. __Change of Metric__: Or realizing that the target metric is different from the current one. There are instances where the entire scope of project changed due to the change in metric.

## References:

1. https://course.fullstackdeeplearning.com/course-content/setting-up-machine-learning-projects/overview
2. https://venturebeat.com/2019/07/19/why-do-87-of-data-science-projects-never-make-it-into-production/
3. https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence

<!-- > “One of the biggest opportunities for all of us today is to figure out how we educate the business leaders across the organization. Before, a leader didn’t need to necessarily know what the data scientist was doing. Now, the data scientist has stepped into the forefront, and it’s actually really important that business leaders understand these concepts.” - [Deborah Leff, CTO for data science and AI at IBM]((https://venturebeat.com/2019/07/19/why-do-87-of-data-science-projects-never-make-it-into-production/)) -->

