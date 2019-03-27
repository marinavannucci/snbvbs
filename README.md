# Scalable Bayesian Variable Selection for Negative Binomial Regression Models

[TOC]

## Introduction
We focus on Bayesian variable selection methods for regression models for count data, and specifically on the negative binomial  linear regression model. We first formulate a Bayesian hierarchical model with a variable selection *spike-and-slab* prior. For posterior inference, we review standard MCMC methods and also investigate a computationally more efficient approach using variational inference.

The negative binomial regression is specified as the following given 


<p align="center"><img alt="$$&#10;\begin{align}&#10;\begin{split}&#10;y_{i} \mid r, \psi_{i} &amp;\sim\text{NB}\left(r,\frac{\exp (\psi_{i})}{1+\exp (\psi_{i})}\right), \\&#10;\psi_{i}  &amp; =\beta_{0}+{\boldsymbol{x}}_{i}^{T}{\boldsymbol{\beta}}. &#10;\end{split}&#10;\end{align}&#10;$$" src="svgs/eb8f89e1e1375c7862f8a0a3cb679979.svg" align="middle" width="238.41509009999996pt" height="64.7419245pt"/></p>

where we consider a sparsity-inducing prior known as the *spike-and-slab* prior and use the following hierarchical priors:

<p align="center"><img alt="$$&#10;\begin{align}&#10;\beta_{k}\mid\gamma_{k} &amp; \sim\gamma_{k}\underbrace{\text{Normal}\left(0,\sigma_{\beta}^{2}\right)}_{\text{slab}}+\left(1-\gamma_{k}\right)\underbrace{\delta_{0}}_{\text{spike}} &amp;&amp; \text{ where }k=\left\{ 1,2,\cdots,p\right\}, \nonumber \\&#10;\gamma_{k} &amp; \sim\text{Bernoulli}\left(\pi\right) &amp;&amp; \text{ where }\pi\in\left[0,1\right],  \nonumber\\&#10;\beta_{0} &amp; \sim \text{Normal}\left(0,\tau_{\beta_{0}}^{-1}\right) &amp;&amp; \text{ where }\tau_{\beta_{0}}^{-1}=\sigma_{\beta_{0}}^{2}, \\&#10;r &amp; \sim\text{Gamma}\left(a_{r},b_{r}\right), \nonumber\\&#10;\sigma_{\beta}^{2}&amp;\sim\text{Scaled-Inv-}\chi^{2}\left(\nu_{0},\sigma_{0}^{2}\right)\nonumber .&#10;\end{align}&#10;&#10;$$" src="svgs/61bea44b8f4e56a5949bee507b0991d0.svg" align="middle" width="497.76412454999996pt" height="160.5608235pt"/></p>



## Install Software
### Prepare environment


### Install GSL
### Install Eigen









