
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from cpolyagamma import parNegBinSSVIIS
import os
import csv
import pickle

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

boston_dataset = load_boston()
scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(boston_dataset.data)
boston = pd.DataFrame(scaled_data, columns=boston_dataset.feature_names)
boston["MEDV"] = boston_dataset.target

## pair-wise correlation range from -0.77 to 0.91
#correlation_matrix = boston.corr().round(2)
## annot = True to print the values inside the square
#sns.heatmap(data=correlation_matrix, annot=True)

X = boston.loc[:, boston.columns != "MEDV"]
Y = boston["MEDV"]
# append noise features
#np.random.seed(2018)
#for i in range(300):
#    X["noise_%02d"%i] = np.random.normal(0, 1, [506, 1])

result = ["seed", "train_rmse", "test_rmse", "train_pearson", "test_pearson", "m", "time"]

# create a folder for api
folder = "%s" % ("ss_viis_boston")
if not os.path.exists(folder):
    os.makedirs(folder)
    print("Directory %s is created." % folder)
else:
    print("Directory %s already exists." % folder)

csvfile = "%s//ss_mcmc_boston.csv" % folder
# split into training and testing
for seed in range(0, 100):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=seed)
    (n, p) = Xtrain.shape

    with open(csvfile, "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(result)

    
    model = parNegBinSSVIIS(seed)
    model.set_data(Xtrain, Ytrain)
    model.set_prior(0.01, 0.01, -1, -50, 30, 1, 10)  # set prior on over-dispersion parameter r
    model.set_viem(100000, 1e-3, 0, 8)
    model.run_viem(False)
    time = model.get_time()

    r = model.get_r()
    beta0 = model.get_beta0()
    omega = model.get_omega_vec()
    beta = model.get_beta_vec()
    pip = model.get_pip_vec()
    sa = model.get_sa()
    gamma = pip > 0.5
    m = sum(gamma)

    mu = np.exp(np.matmul(Xtrain, beta * gamma) + beta0 + np.log(r))
    kappa = 1.0 / r
    train_pearson = np.sum(np.power((Ytrain - mu) / np.sqrt(mu * (kappa * mu)), 2))
    print('seed {} train pearson residual is {}'.format(seed, train_pearson))

    mu = np.exp(np.matmul(Xtest, beta * gamma) + beta0 + np.log(r))
    test_pearson = np.sum(np.power((Ytest - mu) / np.sqrt(mu * (kappa * mu)), 2))
    print('seed {} test pearson residual is {}'.format(seed, test_pearson))

    # prediction
    pred = np.exp(np.matmul(Xtrain, beta * gamma)+ beta0 + np.log(r))
    train_rmse = rmse(Ytrain, pred)
    print('seed {} train RMSE is {}'.format(seed, train_rmse))

    pred = np.exp(np.matmul(Xtest, beta * gamma)+ beta0 + np.log(r))
    test_rmse = rmse(Ytest, pred)
    print('seed {} test RMSE is {}'.format(seed, test_rmse))

    result = [seed, train_rmse, test_rmse, train_pearson, test_pearson, m, time]

    neg_bin_simu = {"beta": beta, "gamma": gamma, "beta0": beta0, "r": r, "time": time,
                    "train_rmse": train_rmse, "test_rmse": test_rmse, "train_pearson": train_pearson, "test_pearson": test_pearson,
                    "m": m, "model": X.columns.get_values()[gamma]}
    pickle.dump(neg_bin_simu, open("%s//negbin_ss_viis_boston_%d.pickle" % (folder, seed), "wb"), protocol=2)


with open(csvfile, "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(result)