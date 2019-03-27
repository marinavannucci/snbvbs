# Scalable Bayesian Variable Selection for Negative Binomial Regression Models

[TOC]

## Introduction
We focus on Bayesian variable selection methods for regression models for count data, and specifically on the negative binomial  linear regression model. We first formulate a Bayesian hierarchical model with a variable selection *spike-and-slab* prior. For posterior inference, we review standard MCMC methods and also investigate a computationally more efficient approach using variational inference.

The negative binomial regression is specified as the following given 


<p align="center"><img alt="$$&#10;\begin{align}&#10;\begin{split}&#10;y_{i} \mid r, \psi_{i} &amp;\sim\text{NB}\left(r,\frac{\exp (\psi_{i})}{1+\exp (\psi_{i})}\right), \\&#10;\psi_{i}  &amp; =\beta_{0}+{\boldsymbol{x}}_{i}^{T}{\boldsymbol{\beta}}. &#10;\end{split}&#10;\end{align}&#10;$$" src="svgs/eb8f89e1e1375c7862f8a0a3cb679979.svg" align="middle" width="238.41509009999996pt" height="64.7419245pt"/></p>

where we consider a sparsity-inducing prior known as the *spike-and-slab* prior as,
<p align="center"><img alt="$$&#10;\begin{align}&#10;\beta_{k}\mid\gamma_{k} \sim\gamma_{k}\text{Normal}\left(0,\sigma_{\beta}^{2}\right)+\left(1-\gamma_{k}\right)\delta_{0},\quad k=1,\cdots,p,&#10;\end{align}&#10;$$" src="svgs/77eb4acc988d1e41d7674cd88ec0442c.svg" align="middle" width="401.53105079999995pt" height="20.50407645pt"/></p>
where <img alt="$\gamma_{k}$" src="svgs/0be70542d78d255d114877bcf3e2b091.svg" align="middle" width="15.77667134999999pt" height="14.15524440000002pt"/> is the latent indicator variable of whether the k-th covariate has a nonzero effect on the outcome, <img alt="$\delta_{0}$" src="svgs/e4d57a6b757d7da2ca852e9d5d1ceee6.svg" align="middle" width="13.858486949999989pt" height="22.831056599999986pt"/> is a point mass distribution at 0, and <img alt="$\sigma_{\beta}^{2}$" src="svgs/c5b9a9fd5941f24be0e2dbdae5d496d2.svg" align="middle" width="17.43826424999999pt" height="26.76175259999998pt"/> is the variance of the prior effect size.


## Install Software
### Prepare environment


### Install GSL
### Install Eigen









