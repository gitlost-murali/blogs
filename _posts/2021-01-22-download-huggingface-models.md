---
title: \[Shorts-1\] How to download HuggingFace models the right way
excerpt: Downloading huggingface models as git repos
tags: [NLP, Deep Learning, HuggingFace, Shorts]
date: 2021-01-22 06:58:10 +0530
categories: nlp huggingface
permalink: /:categories/:title
---

While downloading HuggingFace may seem trivial, I found that a few in my circle couldn't figure how to download huggingface-models. There are others who download it using the "download" link but they'd lose out on the model versioning support by HuggingFace. This micro-blog/post is for them.

### Steps 

* Directly head to [HuggingFace page](https://huggingface.co/) and click on "models".

{% include figure image_path="/assets/images/huggingface-opening.png" alt="HuggingFace landing page" caption="__Figure 1:__ _HuggingFace landing page_."%}

* Select a model. For now, let's select `bert-base-uncased`

{% include figure image_path="/assets/images/hf-models.png" alt="HuggingFace models page" caption="__Figure 2:__ _HuggingFace models page_."%}

* You just have to copy the model link. In our case, https://huggingface.co/bert-base-uncased

_Note:_ Model versioning is done here with the help of [GitLFS](https://git-lfs.github.com/) (Git for Large File Storage). If you haven't already installed it, install it from [here](https://git-lfs.github.com/).

1. Navigate to the directory you want.
2. Type the following commands:
```
git lfs install # Initiate the git LFS system
git clone https://huggingface.co/bert-base-uncased
```

That's it. Your model will be downloaded like a git code repo.