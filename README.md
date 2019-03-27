# Scalable Bayesian Variable Selection for Negative Binomial Regression Models

## Introduction
We focus on Bayesian variable selection methods for regression models for count data, and specifically on the negative binomial  linear regression model. We first formulate a Bayesian hierarchical model with a variable selection *spike-and-slab* prior. For posterior inference, we review standard MCMC methods and also investigate a computationally more efficient approach using variational inference.

The negative binomial regression is specified as the following given 

<p align="center"><img alt="$$&#10;\begin{align}&#10;\begin{split}&#10;y_{i} \mid r, \psi_{i} &amp;\sim\text{NB}\left(r,\frac{\exp (\psi_{i})}{1+\exp (\psi_{i})}\right), \\&#10;\psi_{i}  &amp; =\beta_{0}+{\boldsymbol{x}}_{i}^{T}{\boldsymbol{\beta}}. &#10;\end{split}&#10;\end{align}&#10;$$" src="svgs/eb8f89e1e1375c7862f8a0a3cb679979.svg" align="middle" width="238.41509009999996pt" height="64.7419245pt"/></p>

We implemented the sparsity-inducing *spike-and-slab* prior and adaptive shrinkage horseshoe prior.

**The hierarchical priors are specified for the *spike-and-slab* setting:**
<p align="center"><img alt="$$&#10;\begin{align}&#10;\beta_{k}\mid\gamma_{k} &amp; \sim\gamma_{k}\underbrace{\text{Normal}\left(0,\sigma_{\beta}^{2}\right)}_{\text{slab}}+\left(1-\gamma_{k}\right)\underbrace{\delta_{0}}_{\text{spike}} &amp;&amp; \text{ where }k=\left\{ 1,2,\cdots,p\right\}, \nonumber \\&#10;\gamma_{k} &amp; \sim\text{Bernoulli}\left(\pi\right) &amp;&amp; \text{ where }\pi\in\left[0,1\right],  \nonumber\\&#10;\end{align}&#10;$$" src="svgs/c1224c6612054dbc66c0a66ddee01018.svg" align="middle" width="497.76412454999996pt" height="69.20710005pt"/></p>

**The hierarchical priors are specified for the *horseshoe* setting:**
<p align="center"><img alt="$$&#10;\begin{align}&#10;\left[\beta_{k}\mid\lambda_{k}\right] &amp; \overset{\text{indep}}{\sim}\text{Normal}\left(0,\lambda_{k}^{2}\right)\nonumber \\&#10;\left[\lambda_{k}\mid A\right] &amp; \overset{\text{iid}}{\sim}C^{+}\left(0,A\right)\nonumber\\&#10;A &amp; \sim\text{Uniform}\left(0,10\right)\nonumber &#10;\end{align}&#10;$$" src="svgs/812c9e17e1a2f268405f9b33dea1fa68.svg" align="middle" width="202.8539766pt" height="78.2990802pt"/></p>

**The priors on the other parameters are given as**
<p align="center"><img alt="$$&#10;\begin{align}&#10;\beta_{0} &amp; \sim \text{Normal}\left(0,\tau_{\beta_{0}}^{-1}\right) &amp;&amp; \text{ where }\tau_{\beta_{0}}^{-1}=\sigma_{\beta_{0}}^{2}, \\&#10;r &amp; \sim\text{Gamma}\left(a_{r},b_{r}\right), \nonumber\\&#10;\sigma_{\beta}^{2}&amp;\sim\text{Scaled-Inv-}\chi^{2}\left(\nu_{0},\sigma_{0}^{2}\right)\nonumber.&#10;\end{align}&#10;$$" src="svgs/322705e6cc02ea0b930348f40cbc82aa.svg" align="middle" width="349.73410064999996pt" height="82.97011964999999pt"/></p>

**The direct graph under the *spike-and-slab* setting is** 

<img src="svgs/NegBinGraph.png" alt="drawing" width="500" align="center">



## Installation
### Prepare environment

Our code is written in C/C++ and use the pybind11 to expose the C++ to Python. We use the Eigen and GSL packages for fast linear algebra and random sampling computation. Therefore, to use our code, we need to install a few dependent software and set up the necessary environment. Here is a list of software that need to install before running our code. 

- [x] Python 2 or 3
- [x] [pybind11](https://github.com/pybind/pybind11)
- [x] g++
- [x] git
- [x] cmake
- [x] [GSL](https://www.gnu.org/software/gsl/)
- [x] [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

Configuring those can be a pain but don't worry. I will walk you through the process in detail. For the following tutorial, I will use **Ubuntu** as an example. For the other Linux based system such as macOS or Red Hat, the process should be similar. But the process for setting up the environment in Windows is more confusing and you might need to install [Microsoft Visual Studio](https://visualstudio.microsoft.com/) and its package manager [vcpkg](https://github.com/Microsoft/vcpkg). However, I am currently working on an R package that can hopefully solve the compatibility issues with the Windows. Please visit us again and check our R package in the future.

**Install Python**

You can install either Python 2 or Python 3 using [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 

**Install the pybind11**

```shell
pip install pybind11
```
**Install gcc and g++**
```shell
sudo apt-get install build-essential
```
**Install git**
```shell
sudo apt install git
```
**Install cmake** Check the answer [here](https://askubuntu.com/questions/610291/how-to-install-cmake-3-2-on-ubuntu) for an alternative way.
```shell
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:george-edison55/cmake-3.x
sudo apt-get update
sudo apt-get install cmake
```
**Install GSL**

```shell
sudo apt-get install libgsl-dev
```

**Install Eigen**

```shell
sudo apt-get install libeigen3-dev
```

**Compile csnbvbs**

```shell
cmake . -DCMAKE_BUILD_TYPE=Release
make
```
You will find a file named **csnbvbs.cpython-36m-x86_64-linux-gnu.so ** in the current directory. 
Copy the **csnbvbs**file  to the directory that you are working with. You are ready to import it as a regular python package. Congratulations!

## Example

There are four modules from this package:

- [x] NegBinHS (MCMC version with the horseshoe prior)
- [x] NegBinSSMCMC (MCMC version with the *spike-and-slab* prior)
- [x] NegBinSSVIEM (Variaional Inference EM with the *spike-and-slab* prior)
- [x] parNegBinSSVIIS (Variaional Inference EM and Importance Sampling with the *spike-and-slab* prior)

You can copy the **csnbvbs** file  to the scripts folder and perform the following benchmark test:

Simulate n = 100, p = 50 with various model sparsity and correlation measure rho.
```bash
python py_simulation.py
```
After running the above, you will find a simulation folder with 100 examples per rho. You then can perform benchmark testing among the four methods by 


```shell
python py_ss_mcmc_benchmark.py # spike and spike mcmc sampling
python py_hs_benchmark.py      # horseshoe mcmc sampling
pyhton py_ss_viem_benchmark.py # variational inference EM
pyhton py_ss_viss_benchmark.py # variational inference EM with importance smapling
```
