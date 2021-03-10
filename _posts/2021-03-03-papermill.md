---
title: \[Shorts-2\] Papermill=>Adding Parameters to Python Notebooks & executing them like a function
excerpt: Make Notebooks work like functions
tags: [Python, Shorts, Jupyter Notebook, Notebook, iPython Notebook, Tools]
date: 2021-03-03 06:58:10 +0530
categories: python tooling
permalink: /:categories/:title
---

Python Notebooks are great when you are experimenting/ideating. You can quickly test your ideas. And before you realize, you'll end-up writing the entire code in a notebook. The biggest pain point is to convert the code spanned over 40-50 cells into a python function for looping it over multiple times. This is where __PaperMill helps__.

Let us understand this with a sample script,

<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill_samplecode.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill_samplecode.png"></a>
    <figcaption><b>Figure 1:</b> <i> Sample Script </i></figcaption>
</figure>

**Problem**
If we want to run the entire script with multiple `names = ["def.csv","ghi.csv","abc.csv"]`, 

1. We will have to push all the code into a function with `name` as the argument.
OR
2. `Restart & Run` the notebook while you change the variable `name` for every file.

**Papermill Solution**

1. Papermill tells you to tag the cells which think you are parameters. You can tag your variables' cell the following way,
<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill-tags.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill-tags.png"></a>
    <figcaption><b>Figure 2:</b> <i> Select the option to tag cells </i></figcaption>
</figure>
<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill-tag1.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill-tag1.png"></a>
    <figcaption><b>Figure 3:</b> <i> Name the cell as `parameters`  </i></figcaption>
</figure>
<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill-tag2.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/papermill-tag2.png"></a>
    <figcaption><b>Figure 4:</b> <i> Cell tagged as `parameters` </i></figcaption>
</figure>

Now, use the following code to execute the notebook with different arguments

```python
import papermill as pm

names = ["abc.csv","bcd.csv","efg.csv"]

for name in names:
    pm.execute_notebook(
       'papermill-in.ipynb', ## input notebook
       f'out_pm_{name}.ipynb', ## output notebook
       parameters=dict(name=name) ## parameters
    )
```

Above code executes the notebooks by injecting parameters. You can look at the `injected parameters` in the output notebooks. For ex, in `out_pm_bcd.csv.ipynb`:
<figure>
    <a href="{{ site.url }}/{{ site.baseurl }}/assets/images/pm-injectedparams.png"><img src="{{ site.url }}/{{ site.baseurl }}/assets/images/pm-injectedparams.png"></a>
    <figcaption><b>Figure 5:</b> <i> Injected Parameters by papermill </i></figcaption>
</figure>
