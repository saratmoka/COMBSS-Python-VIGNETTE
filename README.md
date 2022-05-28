# Note
 If you have download the code before 28 May 2022, please download the current version. We made some updates to COMBSS functions.

# COMBSS-VIGNETTE

Here we provide a first vignette to run COMBSS, a novel algorithm for best subset selection for a linear regression model, using Python. For a vignette that runs COMBSS in R, we refer to https://github.com/benoit-liquet/COMBSS-R-VIGNETTE.


This vignette reproduces some replications from the simulation study presented in our article:

> Moka S, Liquet B, Zhu H, and Muller S (2022). COMBSS: Best Subset Selection via Continuous Optimization. https://arxiv.org/abs/2205.02617, 37 pages.

## Getting Started


In this short vignette we use the following Python packages:

- numpy
- scipy
- sklearn
- sys
- matplotlib

## Download the COMBSS python code

Download the file 'combss_functions_github.py' from [here](/combss_functions_github.py) to your working directory. 

##  COMBSS in a low-dimensional setup

In this example, we consider a dataset with n = 100 samples and p = 20 predictors, of which 10 are active predictors.

This analysis is presented [here](/Low_dimensional_example.ipynb).

##  COMBSS in a high-dimensional setup

In this example, we consider a dataset with n = 100 samples and p = 1000 predictors, of which 10 are active predictors.

This analysis is presented [here](/High_dimensional_example.ipynb).

##  COMBSS in a ultra high-dimensional setup

In this example, we consider a dataset with n = 100 samples and p = 10,000 predictors, of which 3 are active predictors.

This analysis is presented [here](/Ultra_high_dimensional_example.ipynb).
