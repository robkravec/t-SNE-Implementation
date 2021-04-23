# STA 663 Final Project (t-SNE)

Reference paper: "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton

Contributors: Marc Brooks, Rob Kravec, Steven Winter

## Description

This repository contains the source code for a new package, `tsne663`, as well as a reproducible report that satisfies the requirements of the STA 663 project. The code to produce the report is contained in `writeup.ipynb` and is converted into a pdf without visible code using the following terminal command: `jupyter nbconvert --to pdf --no-input writeup.ipynb`

## Package functions

The `tsne663` package contains functions to (1) implement t-SNE and (2) test / visualize t-SNE on simulated data. Below, we provide brief descriptions of the key functions:

- `tsne`: Takes in data matrix (and several optional arguments) and returns low-dimensional representation of data matrix with values stored at each iteration
- `make_two_blobs`, `make_parallel_lines`, `make_two_3d_circles`, `make_trefoil_knot`, `make_springs`: Generate data (according to optional arguments) that are well suited to showcase t-SNE's ability to separate clusters
- `perp_plots`, `step_plots`: Creates plots of t-SNE performed on a single dataset for a set of specified perplexity values or iteration numbers
- `compare_plots`: Creates grid of plots to showcase t-SNE's performance on multiple datasets across a range of perplexity values or iteration numbers
