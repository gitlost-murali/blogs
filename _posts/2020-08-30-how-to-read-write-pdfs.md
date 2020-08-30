---
title: "Reading & Writing to PDFs"
excerpt: "Rating Python PDF libraries by their capability, usability etc."
tags: [Engineering, Tooling, PDF, PDF parsing]
mathjax: true
categories: tooling pdf
---

## Motivation for the Post

PDF extraction is admittedly a tough engineering task. I know people who founded startups offering PDF extraction services and failed. Hey, it's not their fault. Even, __Amazon__ failed to offer the service through its product __Textract__. 

There are some beautiful libraries out there trying to perfect the process as much as possible. However, I didn't find a tool that could help me end-to-end in preserving the structure i.e headings, annotations and writing back to the PDF. I found an ensemble of tools each better at individual tasks but stitching them together is a tough task.

This post helps you understand how PDF works and what you can/cannot do with it. __Note:__ I'll be using PyMuPDF in this post.

## Understanding PDF 

What makes PDF parsing (Portable Document Format) so difficult is its way of storing the data. For instance, do you (90s kids especially) remember cutting celebrities' photos from newspapers and pasting them on a page/chart? PDF is inspired from the same methodology. Every word/character is just pasted on a blank page w.r.t coordinate system. So unlike your HTML/XML format, you can't simply find a table by looking at tags because there are __no tags__. This problem is also reflected & [discussed](https://youtu.be/99A9Fz6uHAA) in the recent efforts of [Camelot](https://camelot-py.readthedocs.io/en/master/) and [Tabula](https://tabula-py.readthedocs.io/en/latest/).

Coming back to the question of how storage is done, PDF maintains two layers - Data Layer and Metadata layer.

{% include figure image_path="/assets/images/pdf-parts.png" alt="PDF layering" caption="__Figure 1:__ _How PDF stores data in layers_." %}

Let's understand it better using [this PDF](http://africau.edu/images/default/sample.pdf).

{% include figure image_path="/assets/images/sample-pdf.png" alt="PDF Sample" caption="__Figure 2:__ _Sample PDF file_." %}

For brevity, narrow down our understanding of Data Layer to Words and Metadata to Annotations/Highlights. Now, I have highlighted some portion of the PDF to see how PDF stores the details.

{% include figure image_path="/assets/images/pdf-highlight.png" alt="Sample Highlight" caption="__Figure 3:__ _Highlighting some portion of PDF_." %}

### 1. Getting words' info

```python
import fitz

pdf_filename = "sample-highlight.pdf"
page_number = 0

doc = fitz.open(f'{pdf_filename}') # Read the file as doc
page1 = doc[page_numer] # Get the page object
page1_info = page1.getText("words") # Get Text from a particular page as a list of words.

print(page1_info)
```

 `.getText("words")` gives word level information i.e `rectangle/bounding-box of the word, word, paragraph #, line #, position in line`

```
[
    (x1, y1, x2, y2, word, paragraph #, line #, word-position)
    (72.02999877929688, 92.64202880859375, 90.91999816894531, 106.38202667236328, 'This', 1, 0, 0),
    (93.69999694824219, 92.64202880859375, 100.91999816894531, 106.38202667236328, 'is', 1, 0, 1),
    (103.69999694824219, 92.64202880859375, 109.25999450683594, 106.38202667236328, 'a', 1, 0, 2),
    (112.03999328613281, 92.64202880859375, 135.3699951171875, 106.38202667236328, 'small', 1, 0, 3),
    (138.14999389648438, 92.64202880859375, 201.50997924804688, 106.38202667236328, 'demonstration', 1, 0, 4),
    (204.28997802734375, 92.64202880859375, 220.969970703125, 106.38202667236328, '.pdf', 1, 0, 5),
    (223.74996948242188, 92.64202880859375, 236.52996826171875, 106.38202667236328, 'file', 1, 0, 6),
    (239.30996704101562, 92.64202880859375, 242.6399688720703, 106.38202667236328, '-', 1, 0, 7),
    ...
]
```

### 2. Getting Highlights

 `.getannots()` gives information about each annotation i.e `Type of annotation - Highlight/Text, Rectangle of the highlight`.

```python
for annot in doc[0].annots():
    print(annot)
    print(annot.info)
    print(annot.rect)
```

Note that this will only give you the rectangle coordinates of the annotation but not the word(s) inside it.

```
'Highlight' annotation on page 0 of sample-highlight.pdf
{'content': '', 'name': '', 'title': 'mano', 'creationDate': '', 'modDate': "D:20200830183034+05'30", 'subject': '', 'id': ''}
Rect(135.17147827148438, 94.25, 204.82852172851562, 107.75)
```

To get the words inside highlighted