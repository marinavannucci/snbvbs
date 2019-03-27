from cpolyagamma import NegBinSSVIIS, parNegBinSSVIIS
import numpy as np
from math import log, log10
import matplotlib.pyplot as plt
import pickle
import os

ran_seed = 1
rho = 0.3
negbin_dict = pickle.load(open("./simulation/rho_%3.2f/negbin_simu_%d_rho_%3.2f.pickle"%(rho, ran_seed, rho), "rb" ))
opt_model = negbin_dict["opt_model"]
print(opt_model)

X = negbin_dict["X"]
y = negbin_dict["y"]

model = parNegBinSSVIIS(ran_seed)
model.set_data(X, y)
model.set_prior(0.01, 0.01, -1, -50, 50, 1, 10)  # set prior on over-dispersion parameter r
model.set_viem(100000, 1e-3, 0, 5)
model.run_viem(False)
print(model.get_time())


m_r = model.get_r()
m_beta0 = model.get_beta0()
m_omega = model.get_omega_vec()
m_beta = model.get_beta_vec()
pip = model.get_pip_vec()
m_sa = model.get_sa()
m_gamma = pip > 0.5
