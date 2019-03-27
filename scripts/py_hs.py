from cpolyagamma import test, digamma, NegBinHS, NegBinSSMCMC
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

ran_seed = 2018
negbin_dict = pickle.load(open(os.path.join("simulation", "negbin_simu_%d.pickle" % (ran_seed)), "rb" ))

X = negbin_dict["X"]
y = negbin_dict["y"]

model = NegBinHS(ran_seed)
model.set_data(X, y)
model.set_prior(0.01, 0.01)  # set prior on over-dispersion parameter r
model.set_mcmc(20000, 3000, 1000)
model.run_mcmc(True)

neg_bin_simu = {"beta": model.get_beta_mat(),
                "eta": model.get_eta_mat(),
                "omega": model.get_omega_mat(),
                "beta0": model.get_beta0_vec(),
                "tau": model.get_tau_vec(),
                "r": model.get_r_vec(),
                "time": model.get_time()}
pickle.dump(neg_bin_simu, open("negbin_hs_mcmc_simu_%d.pickle" % (ran_seed), "wb"), protocol=2)



"""
# get back the samples of interest from the sampler
sbetas_mat = model.get_beta_mat()
setas_mat = model.get_eta_mat()
somegas_mat = model.get_omega_mat()
sbeta0_vec = model.get_beta0_vec()
sr_vec = model.get_r_vec()
stau_vec = model.get_tau_vec()

# save everything in dictionary and pass that to ploting algorithm
sample_dict = {"sbetas": sbetas_mat, "setas": setas_mat, "somegas": somegas_mat, 
               "sbeta0": sbeta0_vec, "sr": sr_vec, "stau": stau_vec,
               "negbin_dict": negbin_dict}
pickle_filename = os.path.join("simulation", "negbin_simu_hs_%d.pickle" % (ran_seed))
pickle.dump(sample_dict, open(pickle_filename, "wb" ))

"""