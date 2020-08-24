---
title: "My experience parsing PDFs"
excerpt: "Rating Python PDF libraries by their capability, usability etc."
tags: [Engineering, Tooling, PDF, PDF parsing]
mathjax: true
categories: tooling pdf
---

## The Struggle

PDF extraction is admittedly a tough engineering task. I know people who founded startups offering PDF extraction services and failed. Hey, it's not their fault. Even, __Amazon__ failed to offer the service through its product __Textract__. 

There are some beautiful libraries out there trying to perfect the process as much as possible. However, I didn't find a tool that could help me end-to-end in preserving the structure i.e headings, annotations and writing back to the PDF. I found an ensemble of tools each better at individual tasks but stitching them together is a tough task.

Consider this post like a survey of tools that could help you in your future projects.

## Reading PDFs

### 1. PDFMiner+

Well, there's a base to every other library and that is PDFMiner. PDFMiner3, PDFMinerSix are the official libraries which extended PDFMiner. Many other libraries are also built on top of it.

1. Generated `XML` file deals at character level. So, this wasn't quite intuitive/informative for me.
2. Generated `HTML` file preserves the structure when displayed, however, there's no easy way to figure out what's a sentence and paragraph.
3. Extracted text doesn't preserve structure.
4. Generated `.tag` file preserves the structure and it's differentiate between a sentence and a paragraph.
5. Tables are not extracted.

### 2. Camelot

Released in 2019. Hyped to be the best tool in extracting tables.

1. Focussed entirely on extracting tables, nothing else.
2. Fails extracting tables that are centered and have 1-2 rows.
3. It can identify those tables with weird border lines but doesn't get the structure right. And fails when space is used.

### 3. PDFPlumber

This package surprised me. This is by far the best in terms of extracting tables, text. It offers a lot of features.

1. Better text extraction than PDFMiner variants.
2. Extracts tables better than Camelot (only incase of smaller & centered tables).
3. Doesn't extract tables if tables have border weird lines instead of plain black lines. Completely skips the table.

## Writing to PDFs

### 4. pdfrw

Didn't explore it.
