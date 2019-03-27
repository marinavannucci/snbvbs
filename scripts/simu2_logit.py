
import numpy as np
from numpy import tanh, exp, diag, ones, zeros
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


def simulation(rng_seed, n, p, tp, rho):
    """
    :param rng_seed: random seed
    :param n:        number of samples
    :param p:        number of features
    :return:         dictionary storing all infos
    """
    print("Simulate negative binomial counts regression\n n=%d, p=%d, rho=%3.2f" % (n, p, rho))
    seed(rng_seed)

    # simulate correlated matrix
    X_mu1 = normal(0, .1, p)
    temp = np.abs(np.repeat([range(1, p+1)], p, axis=0).transpose() - np.repeat([range(1, p+1)], p, axis=0))
    X_Sigma1 = np.power(rho, temp)

    # simulate independent matrix
    X_mu2 = zeros(tp - p)
    X_Sigma2 = diag(ones(tp - p))

    # draw correlated samples from MVN(X_mu1, X_Sigma1)
    X1 = multivariate_normal(X_mu1, X_Sigma1, n)

    # draw independent samples from MVN(X_mu2, X_Sigma2)
    X2 = multivariate_normal(X_mu2, X_Sigma2, n)

    # concatenate X1 and X2
    X = np.hstack((X1, X2))

    # sample from bernoulli and uniform for coefficient beta and model space gamma
    opt_gamma = np.append(binomial(1, 0.15, p), zeros(tp - p))  # model gamma
    opt_beta = np.append(uniform(-2, 2, p), zeros(tp - p))  # model coefficient
    # drop all elements in beta whose absolute value less than 0.5
    opt_gamma *= abs(opt_beta) > 0.5 
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
    tp = 1000
    rho = 0.9
    dir = "./simulation/rho_%3.2f" % rho
    if not os.path.exists(dir):
        os.makedirs(dir)
    for ran_seed in range(1, 100, 1):
        try:
            simulation_dict = simulation(ran_seed, n, p, tp, rho)
        except:
            pass
        if sum(simulation_dict["opt_gamma"]) == 0 or sum(simulation_dict["opt_gamma"]) > 10:
            continue
        print("seed: %d"%ran_seed)
        # dump that into pickle file
        pickle.dump(simulation_dict, open("./simulation/rho_%3.2f/negbin_simu_%d_rho_%3.2f.pickle" % (rho, ran_seed, rho), "wb"), protocol=2)
        #print(simulation_dict["opt_gamma"])
        #print(simulation_dict["opt_beta"])
        print(simulation_dict["opt_model"])
