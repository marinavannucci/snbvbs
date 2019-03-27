import numpy as np
from numpy import tanh, exp, diag, ones
from numpy.random import normal, multivariate_normal, uniform, gamma, poisson, binomial, seed
import pickle
import matplotlib.pyplot as plt
import os


def expect_omega(z, y, r):
    """
    compute expectation of omega given the linear function eta and over-dispersion rate r
    :param z: the linear combination term between x and beta
    :param y:   the count data y
    :param r:   the over-dispersion parameter r
    :return:    the expectation of latent variable omega conditioned on eta
    """
    return ((y + r) * (tanh(z / 2) / (2 * z)))


def sigmoid(eta):
    """
        sigmoid transformation of eta
        :param eta: linear combination between x and beta
        :return: sigmoid transformed of linear term eta
    """
    return 1 / (1 + exp(-eta))


def simulation(rng_seed, n, p, rho):
    """
    :param rng_seed: random seed
    :param n:        number of samples
    :param p:        number of features
    :return:         dictionary storing all infos
    """
    print("Simulate negative binomial counts regression\n n=%d, p=%d, rho=%3.2f" % (n, p, rho))
    seed(rng_seed)

    X_mu = normal(0, .1, p)
    #X_Sigma = diag(ones(p))

    # simulate correlated matrix
    temp = np.abs(np.repeat([range(1, p+1)], p, axis=0).transpose() - np.repeat([range(1, p+1)], p, axis=0))
    X_Sigma = np.power(rho, temp)

    # draw independent samples from MVN(X_mu, X_Sigma)
    X = multivariate_normal(X_mu, X_Sigma, n)

    # sample from bernoulli and uniform for coefficient beta and model space gamma
    opt_gamma = binomial(1, 0.15, p)  # model gamma
    opt_beta = uniform(-2, 2, p)
    # drop all elements in beta whose absolute value less than 0.5
    opt_gamma &= abs(opt_beta) > 0.5 
    opt_beta *= opt_gamma  # coefficients
    opt_beta0 = 2  # bias
    opt_r = 1  # over-dispersion parameter
    opt_z = np.dot(X, opt_beta) + opt_beta0
    opt_lam = gamma(opt_r, exp(opt_z), n)
    y = poisson(opt_lam, n)

    opt_omega = expect_omega(opt_z, y, opt_r)

    # put everything into a dictionary and return back
    negative_binomial_dict = {"X": X, "y": y, "opt_beta": opt_beta, "opt_beta0": opt_beta0,
                              "opt_r": opt_r, "opt_gamma": opt_gamma, "opt_omega": opt_omega, 
                              "opt_m": np.sum(opt_gamma), "opt_model": np.nonzero(opt_gamma),
                              "seed": ran_seed}

    return negative_binomial_dict


if __name__ == "__main__":
    # generate random negative binomial regression
    n = 200
    p = 50
    for rho in [0.0, 0.3, 0.6, 0.9]:
        dir = "./simulation/rho_%3.2f" % rho
        if not os.path.exists(dir):
            os.makedirs(dir)
        for ran_seed in range(100):
            try:
                simulation_dict = simulation(ran_seed, n, p, rho)
            except:
                pass
            if sum(simulation_dict["opt_gamma"]) == 0 or sum(simulation_dict["opt_gamma"]) > 10:
                continue
            print("seed: %d"%ran_seed)
            # dump that into pickle file
            pickle.dump(simulation_dict, open("./simulation/rho_%3.2f/negbin_simu_%d_rho_%3.2f.pickle" % (rho, ran_seed, rho), "wb"))