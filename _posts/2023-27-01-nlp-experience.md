---
title: Delineating my NLP experience
excerpt: Talks about my experience in NLP - tools, techniques, and projects.
tags: [data science, nlp, interviews]
date: 2023-01-27 18:58:10 +0530
categories: python data-science nlp
permalink: /:categories/:title
toc: True
---

## Tools:

* __Foundational NLP tools__: I have used NLTK, Spacy, Gensim, Stanford CoreNLP suite. With the advent of Transformers and rise of pretrained language models, I consider it would be redundant mentioning them but anyway, here we go.
* __Deep Learning Platforms__: I'm a constant advisor of PyTorch. But at the same time, I couldn't avoid mandatory Tensorflow/Keras assignments in my Masters. I use FastAI toolkit for learning purposes.
* __NLP Libraries__: I have used HuggingFace Transformers, AllenNLP, Flair, and OpenNMT. OpenNMT for building machine translation baselines for my undergrad thesis. HuggingFace Transformers is something I use in my most of the recent projects. AllenNLP for coreference resolution and Flair for sentiment analysis and NER.
* __NLP Architectures__: I am up to date with the recent transformer variants and language models (Auto-regressive and Denoising based models) BERT, RoBERTa, DeBerta, XLNet, T5, BART, GPT-1,2,3, CLIP etc. And at the same time, if there is any need to use any of the older techniques, dating back to automata, I can do that too. Here's my [mini blog](https://gitlost-murali.github.io/blogs/python/data-science/nlp/foma) explaining how to use FOMA tool (automata library) for a downstream NLP task.

## Techniques:

* __Machine Translation__: I have built baselines for machine translation using OpenNMT. At the same time, my [thesis](https://gitlost-murali.github.io/blogs/thesis/Undergradthesis_Murali_Manohar.pdf) was on __unsupervised machine translation__. Here, on the top of standard back translation approach, we proposed that leveraging a lexicon built from cross-lingual embeddings would greatly impact the convergence of training. I have also used HuggingFace Transformers for fine-tuning the models for my recent college projects.

Pharamacuetical giant, Novartis, wanted to have a in-house translation service as they didn't want to rely on Google Translate citing privacy concerns. So, I built a streamlit webapp around [HelsinkiNLP](https://github.com/Helsinki-NLP/Opus-MT) models and deployed it on their servers. This was also a great learning experience as I got to learn about the deployment of models in production.

* __Text Classification__: This has been the most popular and common task in NLP. Starting from 2017, I have used different techniques for text classification. I have used CNNs, RNNs, LSTMs, GRUs, Transformers, etc. I have also used different techniques for feature engineering like TF-IDF, Word2Vec, GloVe, FastText, etc.

There has been different variants to it. For example, in a course project of building __stance detection__ in Covid vaccine, we combined Twitter network features with text encoder features. Trust me on this, this is an __interesting read__ (mainly, the user network features computation). Here's the [link](https://gitlost-murali.github.io/blogs/work/Apps_1_NLP_WriteUp.pdf). The professor, [Prof. Rodrigo Agerri](https://ragerri.github.io/) wanted me to publish this work with some tweaks but I was busy shifting to Netherlands for 2nd year of my Masters and couldn't work on it further.

There was also something called [__sentiment neuron__](https://openai.com/blog/unsupervised-sentiment-neuron/) from OpenAI, where they use one node's activation value to decide the sentiment value of text. I built a wrapper around it and passed it to the labeling team in my former company, Gramener.

* __Named Entity Recognition__: The 2nd most used task in NLP. Starting with BiLSTM CRF model as my baseline in 2018, I have now experimented (successfully) pivoting NER task as seq2seq task with generative models like BART, T5, etc. In Industry, while building a patient-deidentification app for Novartis, I have used Spacy, SciSpacy, ClinicalBERT, BioBERT, AllenNLP, Flair etc. 

* __Coreference Resolution__: Again, for the same project in Novartis, we used AllenNLP's models for coreference resolution of patient/subject instances.

* __Question Answering__: In my recent efforts to pivot NER task into multiple other tasks, I have used BERT based model for this question answering task.

* __Math Word Problem Solving__: This is my current thesis. Here, we will be dealing with seq2seq, sequence2tree and graph2tree models, which leverage different attention mechanisms and architectures like Graph Attention Networks, Graph Convolutional Networks, etc. 

* __Unsupervised Word Alignment__: 
I wrote a few project proposals to Netflix and Duolingo based on this idea. Here's a sample image of the application I built for them. Note that there is no training involved. This is completely zero shot.

* __Multi-modal work__: We built a MEME retrieval system using CLIP, sentence embeddings and presented our poster. This is again a __very interesting work__. Here's the [link](https://gitlost-murali.github.io/blogs/work/memeR_poster.pdf). We also built a demo but it's down considering the costs.

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/work/proposal.png"><img src="{{ site.url }}/{{ site.baseurl }}/work/proposal.png"></a>
    <figcaption><b>Figure 1:</b> <i> Proposal image </i></figcaption>
</figure>

## Reading skills:

I believe I should point my ability to read research papers. As a proof, here's the [folder](https://drive.google.com/drive/folders/1gWj4l2rMof41c9vRaI0uMqVcNcMz0NqS?usp=share_link) with all the annotated research papers I read in the last month. Reading 23 research papers (with 4 survey papers) in a month would require some understanding of NLP techniques. I also summarize my papers visually for advisors. An example file is [here](https://drive.google.com/drive/folders/1gWj4l2rMof41c9vRaI0uMqVcNcMz0NqS?usp=share_link)

## Writing skills:

Before starting my masters, I maintained a technical blog where I wrote about my projects. I didn't highlight it in the interview. Here's the [link](https://gitlost-murali.github.io/blogs/). There are [medium posts](https://medium.com/@kmanoharmurali) on [Unsupervised Translation](https://medium.com/@kmanoharmurali/an-overview-of-unsupervised-machine-translation-mt-f3298dcd6206) and [Generative Adversarial Networks (GANs)](https://medium.com/@kmanoharmurali/friendly-introduction-to-gans-357cf0a99a6e).

Here's my latest [course work paper](https://gitlost-murali.github.io/blogs/work/LfD_Final_Project.pdf). You will find it __interesting__. Explainable NLP could have been my thesis topic but I pivoted to Math Word Problem Solving.

__Coding skills:__

My current part time work at AskUI has made my code much more efficient. Here's a [recent repository](https://github.com/gitlost-murali/thesiscode/tree/main/t5-scripts) I am maintaing. You can see the code modularity, unittests and readability. But since the emphasis is on NLP, I will not go into much details.