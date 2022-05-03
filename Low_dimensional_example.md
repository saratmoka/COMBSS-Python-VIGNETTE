Python packages
---------

``` Python
import numpy as np
from numpy.linalg import inv, norm
from scipy.sparse.linalg import cg
import time
from sklearn import metrics
```

Load function useful for COMBSS
-------------------------------

``` python
from combss-functions-github import *
```

Generate data from a true model
-------------------------------

``` python
n = 100
p = 20
beta_type = 1
K0 = 10
beta0, model0 = gen_beta0(p, K0, beta_type)

snr = 6              
rho = 0.5 

meanX = np.zeros(p)
covX = cov_X(p, rho)
noise_var = beta0.T@covX@beta0/snr

set.seed(140)

data_train = gen_Data(n, p, meanX, covX, noise_var, beta0)
X_train = data[0]
y_train = data[0]
```

### Generation of a validation set

``` python
n_test = 5000

data_test = gen_Data(n_test, p, meanX, covX, noise_var, beta0)
X_test = data_test[0]
y_test = data_test[0]
```

Parameters for COMBSS
--------------------

``` r
CG <- TRUE
alpha <- 0.1
Niter <- 1000
epoch <- 10
tol <- 0.001
tau=0.5
epoch <- 10
trunc <- 0.001
```

Grid of lambda values
---------------------

``` r
lambda.max <- sum(y*y)/n
c.grid <- 0.8
nlambda <- 50
grid.comb <- rev(lambda.max*(c.grid^(c(0:(nlambda-1)))))
```

Compuation of the MSE on the validation set
-------------------------------------------

``` r
mse <- rep(0,nlambda)
nsel <- rep(0,nlambda)
for(j in 1:nlambda){
#print(j)
lam <- as.numeric(grid.comb[j])
model.combssR <- ADAM.COMBSS(X,y,delta=dim(X)[1],lambda=lam,tau=tau,Niter=Niter,alpha=alpha,epoch=epoch,tol=tol,CG=CG,trunc=trunc)
nsel[j] <- sum(model.combssR$s)
y.pred <- as.vector(predict.COMBSS(model.combssR,Xtest))

if(sum(model.combssR$s)>n){mse[j]<- 9999}else{
  mse[j] <- mean((ytest-y.pred)**2)}
}
```

Choice of lambda based on the MSE from the validation set
---------------------------------------------------------

``` r
lambda.min <- grid.comb[which.min(mse)]
plot(mse~log(grid.comb),type="o",col="red",pch=20,xlab=expression(log(lambda)),ylab="MSE (validation set)")
axis(side=3,at=log(grid.comb),labels=paste(nsel),tick=FALSE,line=0)
abline(v=log(lambda.min),lty=3)
```

![](Low_dimensional_example_files/figure-markdown_github/unnamed-chunk-87-1.png)

Run COMBSS with best lambda
---------------------------

``` r
model.combssF <- ADAM.COMBSS(X,y,delta=dim(X)[1],lambda=lambda.min,tau=tau,Niter=Niter,alpha=alpha,epoch=epoch,tol=tol,CG=CG,trunc=trunc)
```

Confusion matrix
----------------

``` r
Selected <- model.combssF$s
True <- as.logical(beta)
table(True,Selected)
```

    ##        Selected
    ## True    FALSE TRUE
    ##   FALSE    10    0
    ##   TRUE      0   10
