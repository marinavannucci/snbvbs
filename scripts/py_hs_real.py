import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from cpolyagamma import NegBinHS
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
np.random.seed(2018)
for i in range(300):
    X["noise_%02d"%i] = np.random.normal(0, 1, [506, 1])

result = ["seed", "train_rmse", "test_rmse", "train_pearson", "test_pearson", "m", "time"]

# create a folder for api
folder = "%s" % ("hs_mcmc_boston")
if not os.path.exists(folder):
    os.makedirs(folder)
    print("Directory %s is created." % folder)
else:
    print("Directory %s already exists." % folder)

csvfile = "%s//hs_mcmc_boston.csv" % folder
# split into training and testing
for seed in range(0, 100):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=seed)
    (n, p) = Xtrain.shape

    with open(csvfile, "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(result)

    model = NegBinHS(seed)
    model.set_data(Xtrain, Ytrain)
    model.set_prior(0.01, 0.01)  # set prior on over-dispersion parameter r
    model.set_mcmc(10000, 3000, 1000)
    model.run_mcmc(True)

    time = model.get_time()
 
    #tau = np.percentile(model.get_eta_mat(), .5)
    tau = 100
    beta_mat = model.get_beta_mat() * (model.get_eta_mat() < tau)
    beta0_vec = model.get_beta0_vec()
    r_vec = model.get_r_vec()
    eta = np.median(model.get_eta_mat(), axis=0)
    
    gamma = eta < tau
    beta = np.median(beta_mat, 0)
    beta0 = np.median(beta0_vec, 0)
    r = np.median(r_vec)
    m = sum(gamma)

    mu = np.mean(np.exp(np.matmul(Xtrain, beta_mat.transpose()) + beta0_vec + np.log(r_vec)), 1)
    kappa = np.mean(1.0 / r_vec)
    train_pearson = np.sum(np.power((Ytrain - mu) / np.sqrt(mu * (kappa * mu)), 2))
    print('seed {} train pearson residual is {}'.format(seed, train_pearson))

    mu = np.mean(np.exp(np.matmul(Xtest, beta_mat.transpose()) + beta0_vec + np.log(r_vec)), 1)
    kappa = np.mean(1.0 / r_vec)
    test_pearson = np.sum(np.power((Ytest - mu) / np.sqrt(mu * (kappa * mu)), 2))
    print('seed {} test pearson residual is {}'.format(seed, test_pearson))

    # prediction
    pred = np.exp(np.matmul(Xtrain, beta)+ beta0 + np.log(r))
    train_rmse = rmse(Ytrain, pred)
    print('seed {} train RMSE is {}'.format(seed, train_rmse))

    pred = np.exp(np.matmul(Xtest, beta)+ beta0 + np.log(r))
    test_rmse = rmse(Ytest, pred)
    print('seed {} test RMSE is {}'.format(seed, test_rmse))

    result = [seed, train_rmse, test_rmse, train_pearson, test_pearson, m, time]

    neg_bin_simu = {"beta": beta, "eta": eta, "beta0": beta0, "tau": tau, "r": r, "time": time,
                    "train_rmse": train_rmse, "test_rmse": test_rmse, "train_pearson": train_pearson, "test_pearson": test_pearson,
                    "m": m, "model": X.columns.get_values()[gamma],
                    "beta_mat": beta_mat, "r_vec": r_vec, "beta0_vec": beta0_vec}
    pickle.dump(neg_bin_simu, open("%s//negbin_hs_mcmc_boston_%d.pickle" % (folder, seed), "wb"), protocol=2)


with open(csvfile, "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(result)