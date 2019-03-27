

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cpolyagamma import parNegBinSSVIIS
import os
import csv
import pickle
import glob


# create a folder for api
folder = "%s" % ("jie_viis_16")
if not os.path.exists(folder):
    os.makedirs(folder)
    print("Directory %s is created." % folder)
else:
    print("Directory %s already exists." % folder)

csvfile = "%s//jie_viem.csv" % folder
result = ["gene", "r", "beta0"] + ["sample_%02d"%i for i in range(15)]  + ["ctrlCoef", "ctrlEffect"]

for filename in glob.glob("./rawdata/*.csv"):
#filename = "./rawdata/ENSG00000281103_1.csv"
    dat = pd.read_csv(filename)

    with open(csvfile, "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(result)

    X = dat.iloc[:, 1:].values
    y = dat.iloc[:, 0].values
    (n, p) = X.shape

    model = parNegBinSSVIIS(2018)
    model.set_data(X, y)
    model.set_prior(0.01, 0.01, -1, -50, 30, 1, 10)  # set prior on over-dispersion parameter r
    model.set_viem(100000, 1e-4, 0, 8)
    model.run_viem(False)
    time = model.get_time()

    r = model.get_r()
    beta0 = model.get_beta0()
    beta = model.get_beta_vec()
    pip = model.get_pip_vec()
    sa = model.get_sa()
    gamma = pip > 0.5
    m = sum(gamma)
    gene = os.path.basename(filename).split(os.extsep)[0]

    negbin_dict = {"gene": gene,
                   "ctrlCoef": beta[-1], "ctrlEffect": gamma[-1],
                   "r": r, "beta0": beta0, "beta": beta,
                   "gamma": gamma, "pip": pip, "sa": sa, "time": time}

    pickle.dump(negbin_dict, open("%s//viem_%s.pickle" % (folder, gene), "wb"), protocol=2)

    result = [gene, r, beta0] + list(beta * gamma) + [pip[-1]]

with open(csvfile, "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(result)
