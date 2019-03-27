
from cpolyagamma import NegBinSSVIEM
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

ran_seed = 81
rho = 0
negbin_dict = pickle.load(open("./simulation/rho_%3.2f/negbin_simu_%d_rho_%3.2f.pickle"%(rho, ran_seed, rho), "rb" ))

X = negbin_dict["X"]
y = negbin_dict["y"]
opt_model = negbin_dict["opt_model"]
opt_omega = negbin_dict["opt_omega"]
opt_beta = negbin_dict["opt_beta"]
opt_gamma = negbin_dict["opt_gamma"] == 1
print("Simulation seed is %d"%ran_seed)
print("Optimal model is")
print(opt_model)
print("true beta coefficient is: ")
print(opt_beta[opt_gamma])

n, p = X.shape
m = sum(negbin_dict["opt_gamma"])

model = NegBinSSVIEM(ran_seed)
model.set_data(X, y)
model.set_prior(0.01, 0.01, m, p - m, 1, 10)  # set prior on over-dispersion parameter r
model.set_viem(100000, 1e-5, 2)
model.run_viem(True)

m_r = model.get_r()
m_beta0 = model.get_beta0()
m_omega = model.get_omega_vec()
m_beta = model.get_beta_vec()
pip = model.get_alpha_vec()
m_sa = model.get_sa()
m_gamma = pip > 0.5

print("time spent in fitting model is %3.2f" % model.get_time())

plt.plot(model.get_elbo_vec())
plt.show()
