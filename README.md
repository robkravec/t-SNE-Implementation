# Spring 2021 STA 663 Final Project (t-SNE)

Reference paper: "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton

Contributors: Marc Brooks, Rob Kravec, Steven Winter

## Description

This repository contains the source code for a new package, `tsne663`, as well as a reproducible report that satisfies the requirements of the STA 663 project. Producing `Implementing t-SNE in Python with Optimized Code and Examples.pdf` requires multiple steps, which include downloading data, generating and saving plots, and running code within `Implementing t-SNE in Python with Optimized Code and Examples.ipynb`, all of which take a non-negligible amount of time. To make this process more modular, we use a `Makefile`, and `Implementing t-SNE in Python with Optimized Code and Examples.pdf` can be reproduced by simply navigating to this repository in the terminal and typing `make`. For aesthetics, we do not show any code in `Implementing t-SNE in Python with Optimized Code and Examples.pdf`, but all code can be seen in `Implementing t-SNE in Python with Optimized Code and Examples.ipynb` and the `.py` files referenced in the `Makefile`.

## Installation

This package (`tsne663`) can be installed from [TestPyPI](https://test.pypi.org/project/tsne663/1.0/) using `pip install -i https://test.pypi.org/simple/ tsne663==1.0`. Alternatively, one can navigate to this Github repository and run `python setup.py install`. 

Upon installation, the following prerequisites will also be installed (versions shown are those used during development, which ensures that the package will work properly):

- `matplotlib` - 3.1.1
- `numpy` - 1.18.1
- `numba` - 0.48.0
- `tqdm` - 4.42.0
- `sklearn` - 0.22.1

Two additional packages, while not automatically installed, are necessary to reproduce the results from the simulations:

- `plotly` - 4.5.0
- `textwrap` - 3.9.4

## Package functions

The `tsne663` package contains functions to (1) implement t-SNE and (2) test / visualize t-SNE on simulated data. Below, we provide brief descriptions of the key functions:

- `tsne`: Takes in data matrix (and several optional arguments) and returns low-dimensional representation of data matrix with values stored at each iteration
- `make_two_blobs`, `make_parallel_lines`, `make_two_3d_circles`, `make_trefoil_knot`, `make_springs`: Generate data (according to optional arguments) that are well suited to showcase t-SNE's ability to separate clusters
- `perp_plots`, `step_plots`: Creates plots of t-SNE performed on a single dataset for a set of specified perplexity values or iteration numbers
- `compare_plots`: Creates grid of plots to showcase t-SNE's performance on multiple datasets across a range of perplexity values or iteration numbers
