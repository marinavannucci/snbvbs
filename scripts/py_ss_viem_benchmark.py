from csnbvbs import test, NegBinSSVIEM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from metrics import comp_performance
import numpy as np
import pickle
import os
import glob

# compute the mean and std of the ROC curve
tprs = []
aucs = []
times = []
mccs = []
precisions = []
recalls = []
F1s = []
mean_fpr = np.linspace(0, 1, 100)
rhos = [0.0, 0.3]
method = "ss_viem"

for rho in rhos:
    # create benchmark folder
    dir = "./benchmark/%s_rho_%3.2f" % (method, rho)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in glob.glob("./simulation/rho_%3.2f/*.pickle"%rho)[0:100]:
        negbin_dict = pickle.load(open(file, "rb" ))
        X = negbin_dict["X"]
        y = negbin_dict["y"]
        ran_seed = negbin_dict["seed"]

        opt_omega = negbin_dict["opt_omega"]
        opt_beta = negbin_dict["opt_beta"]
        opt_gamma = negbin_dict["opt_gamma"] == 1
        n, p = X.shape
        m = sum(opt_gamma)

        print("Number of true variables is %d"%m)
        print("true model is: ")
        print(np.nonzero(opt_gamma))
        print("true beta coefficient is: ")
        print(opt_beta[opt_gamma])


        model = NegBinSSVIEM(ran_seed)
        model.set_data(X, y)
        model.set_prior(0.01, 0.01, m, 2 * (p - m), 1, 10)  # set prior on over-dispersion parameter r
        model.set_viem(100000, 1e-3, 0)
        model.run_viem(True)

        m_r = model.get_r()
        m_beta0 = model.get_beta0()
        m_omega = model.get_omega_vec()
        m_beta = model.get_beta_vec()
        pip = model.get_alpha_vec()
        m_sa = model.get_sa()
        m_gamma = pip > 0.5

        # plot ROC and compute AUC
        fpr, tpr, thresholds = roc_curve(opt_gamma, pip)
        roc_auc = auc(fpr, tpr)

        neg_bin_simu = {"seed": ran_seed, "beta": m_beta, "pip": pip,
                        "beta0": m_beta0, "sa": m_sa,
                        "r": m_r, "time_spent": model.get_time(), 
                        "performance": comp_performance(m_gamma, opt_gamma), 
                        "auc": roc_auc}
        print(comp_performance(m_gamma, opt_gamma))

        # save performance
        times.append(neg_bin_simu["time_spent"])
        mccs.append(neg_bin_simu["performance"]["MCC"])
        precisions.append(neg_bin_simu["performance"]["precision"])
        recalls.append(neg_bin_simu["performance"]["recall"])
        F1s.append(neg_bin_simu["performance"]["F1"])


        # now plot the ROC curve
        tpr = np.append(0, tpr)
        fpr = np.append(0, fpr)
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        plt.plot(fpr, tpr, lw=1, alpha=1.0, label='Simulation %d (AUC = %0.2f)' % (ran_seed, roc_auc))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for HS simulation %d'%ran_seed)
        plt.legend(loc="lower right")
        fig.savefig(os.path.join(dir, "negbin_%s_seed_%d_%3.2f.pdf"%(method, ran_seed, rho)), bbox_inches='tight')
        plt.close()

        # collect ROC statistics
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        pickle.dump(neg_bin_simu, open(os.path.join(dir, "negbin_%s_simu_%d_rho_%3.2f.pickle" % (method, ran_seed, rho)), "wb"), protocol=2)
    # save all results into a dictionary
    pickle.dump({"tprs": tprs, "aucs": aucs, "times": times, "mccs": mccs, "precisions": precisions, "recalls": recalls, "F1s": F1s}, 
                open(os.path.join(dir, "negbin_%s_rho_%3.2f_summary.pickle"%(method, rho)), "wb"), protocol=2)