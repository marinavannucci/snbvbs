import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

ran_seed = 2018
method = "hs"
neg_bin_dict = pickle.load(open(os.path.join("simulation", "negbin_simu_%s_%d.pickle" % (method, ran_seed)), "rb"))

opt_beta = neg_bin_dict["negbin_dict"]["opt_beta"]
opt_beta = neg_bin_dict["negbin_dict"]["opt_beta"]
opt_beta0 = neg_bin_dict["negbin_dict"]["opt_beta0"]
opt_r = neg_bin_dict["negbin_dict"]["opt_r"]
opt_omega = neg_bin_dict["negbin_dict"]["opt_omega"]
opt_gamma = neg_bin_dict["negbin_dict"]["opt_gamma"]
X = neg_bin_dict["negbin_dict"]["X"]
y = neg_bin_dict["negbin_dict"]["y"]

sbetas = neg_bin_dict["sbetas"]
beta0s = neg_bin_dict["sbeta0"]
rs = neg_bin_dict["sr"]
somegas = neg_bin_dict["somegas"]

# compute posterior mean
mbeta = sbetas.mean(axis=0)
momega = somegas.mean(axis=0)

# plt.plot(mbeta, opt_beta, 'ro')
plt.plot(momega, opt_omega, 'ro')
plt.show()

plt.hist(beta0s, 30)
plt.show()

plt.hist(rs, 30)
plt.show()