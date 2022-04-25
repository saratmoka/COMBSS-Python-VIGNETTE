# COMBSS-VIGNETTE

Here we provide a first vignette to run the COMBSS algorithm for best subset selection for a linear regression model.

Here we use Python code. 


This vignette reproduces some replications from the simulation study presented in our article :

> Moka S, Liquet B, Zhu H, and Muller S (2022). COMBSS: Best Subset Selection via Continuous Optimization. *Submitted to arXiv*, 36 pages.


## Getting started

In this short vignette we use the following Python packages

``` Python
numpy
scipy
time
sklearn
```

##  COMBSS in a low dimensional data context

- We consider in this simulated example, a data set of n=100 samples and p=20 predictors, and 10 are active predictors.

- This analysis is presented [here](/Low_dimensional_example.md)

 
## COMBSS in a high dimensional data context

- We consider in this simulated example, a data set of n=100 samples and p=1000 predictors and 10 active predictors.

- This analysis is presented [here](/High_dimensional_example.md)


## COMBSS in a ultra high dimensional data context

- We consider in this simulated example, a data set of n=100 samples and p=10,000 predictors, and only 3 are active predictors.

- This analysis is presented [here](/Ultra_High_dimensional_example.md)
