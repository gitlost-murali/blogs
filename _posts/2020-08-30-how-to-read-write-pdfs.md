---
title: "Reading & Writing to PDFs"
excerpt: "Rating Python PDF libraries by their capability, usability etc."
tags: [Engineering, Tooling, PDF, PDF parsing]
mathjax: true
categories: tooling pdf
---

## Motivation for the Post

PDF extraction is admittedly a tough engineering task. I know people who founded startups offering PDF extraction services and failed. Hey, it's not their fault. Even, __Amazon__ failed to offer the service through its product __Textract__. 

There are some beautiful libraries out there trying to perfect the process as much as possible. However, I didn't find a tool that could help me end-to-end in preserving the structure i.e headings, annotations and writing back to the PDF. I found an ensemble of tools each better at individual tasks but stitching them together is a tough task. After dedicating good amount of time, I was able to zero-down all my requirements to one tool albeit it requires some coding for you.

So, this post helps you understand how PDF works and what you can/cannot do with it. __Note:__ I'll be using PyMuPDF in this post.

## Understanding PDF 

What makes PDF parsing (Portable Document Format) so difficult is its way of storing the data. For instance, do you (90s kids especially) remember cutting celebrities' photos from newspapers and pasting them on a page/chart? PDF is inspired from the same methodology. Every word/character is just pasted on a blank page w.r.t coordinate system. So unlike your HTML/XML format, you can't simply find a table by looking at tags because there are __no tags__. This problem is also reflected & [discussed](https://youtu.be/99A9Fz6uHAA) in the recent efforts of [Camelot](https://camelot-py.readthedocs.io/en/master/) and [Tabula](https://tabula-py.readthedocs.io/en/latest/).

Coming back to the question of how storage is done, it is __rectangles__. Yes, we'll get back to why that's the case in just a minute. Also, note that PDF maintains two layers - Data Layer and Metadata layer for storing different types of data.

{% include figure image_path="/assets/images/pdf-parts.png" alt="PDF layering" caption="__Figure 1:__ _How PDF stores data in layers_." %}

Let's understand it better using [this PDF](http://africau.edu/images/default/sample.pdf).

{% include figure image_path="/assets/images/sample-pdf.png" alt="PDF Sample" caption="__Figure 2:__ _Sample PDF file_." %}

For brevity, narrow down our understanding of Data Layer to Words and Metadata to Annotations/Highlights. Now, I have highlighted some portion of the PDF to see how PDF stores the details.

{% include figure image_path="/assets/images/pdf-highlight.png" alt="Sample Highlight" caption="__Figure 3:__ _Highlighting some portion of PDF_." %}

### 1. Reading
#### 1.1 Getting words' info

```python
import fitz

pdf_filename = "sample-highlight.pdf"
page_number = 0

doc = fitz.open(f'{pdf_filename}') # Read the file as doc
page1 = doc[page_numer] # Get the page object
page1_info = page1.getText("words") # Get Text from a particular page as a list of words.

print(page1_info)
```

 `.getText("words")` gives word-level information i.e `[rectangle/bounding-box of the word, word, paragraph #, line #, position in line]`

```
[
    (x1, y1, x2, y2, word, paragraph #, line #, word-position)
    (72.02, 92.64, 90.91, 106.38, 'This', 1, 0, 0),
    (93.69, 92.64, 100.91, 106.38, 'is', 1, 0, 1),
    (103.69, 92.64, 109.25, 106.38, 'a', 1, 0, 2),
    (112.03, 92.64, 135.36, 106.38, 'small', 1, 0, 3),
    (138.14, 92.64, 201.50, 106.38, 'demonstration', 1, 0, 4),
    (204.28, 92.64, 220.96, 106.38, '.pdf', 1, 0, 5),
    (223.74, 92.64, 236.52, 106.38, 'file', 1, 0, 6),
    (239.30, 92.64, 242.63, 106.38, '-', 1, 0, 7),
    ...
]
```

#### 1.2. Getting Highlights

 `.getannots()` gives information about each annotation i.e `[Type of annotation - Highlight/Text, Rectangle of the highlight]`.

```python
for annot in doc[0].annots():
    print(annot)
    print(annot.info)
    print(annot.rect)
```

Note that this will only give you the rectangle coordinates of the annotation/highlight but not the word(s) inside it because of the storage structure.

```
'Highlight' annotation on page 0 of sample-highlight.pdf
{'content': '', 'name': '', 'title': 'mano', 'creationDate': '', 'modDate': "D:20200830183034+05'30", 'subject': '', 'id': ''}
Rect(135.17, 94.25, 204.82, 107.75)
```

Glimpse of how words & annotations are stored inside PDF.

<figure class="half">
    <a href="{{ site.url }}{{ site.baseurl }}/assets/images/pdf-text-only.png"><img src="{{ site.url }}{{ site.baseurl }}/assets/images/pdf-text-only.png"></a>
    <a href="{{ site.url }}{{ site.baseurl }}/assets/images/pdf-only-highlight.png"><img src="{{ site.url }}{{ site.baseurl }}/assets/images/pdf-only-highlight.png"></a>
    <figcaption><b>Figure</b>: <i>Left</i>: Data layer of PDF. && <i>Right</i>: Annotation layer of PDF.</figcaption>
</figure>

To get the words inside highlighted, we need to map rectangle-coordinates of either sides. In our case, rectangle coordinates for
```
word "demonstration"  is  138.14, 92.64, 201.50, 106.38
highlighted area      is  135.17, 94.25, 204.82, 107.75
```
So, one can write a simple script to find what words coincide-with/lie-inside the annotation/highlight.

### 2. Writing to PDFs

Here comes the most important question, how do you write to PDFs? Again, it's __rectangles__.