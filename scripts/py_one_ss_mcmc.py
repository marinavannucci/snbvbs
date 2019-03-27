import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from cpolyagamma import NegBinSSMCMC

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class App(object):
    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:]) # a dictionary of opts
        self.main(opts)

    def real_data(self, flag):
        # flag large or small
        boston_dataset = load_boston()
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(boston_dataset.data)
        boston = pd.DataFrame(scaled_data, columns=boston_dataset.feature_names)
        boston["MEDV"] = boston_dataset.target

        self.X = boston.loc[:, boston.columns != "MEDV"]
        self.Y = boston["MEDV"]
        
        # append noise features
        np.random.seed(2018)
        if flag == 1:
            self.folder = "%s" % ("hs_mcmc_boston_large")
            for i in range(300):
                self.X["noise_%03d"%i] = np.random.normal(0, 1, [506, 1])
        else:
            self.folder = "%s" % ("hs_mcmc_boston")
        
        # create a folder for api
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            print("Directory %s is created." % self.folder)
        else:
            print("Directory %s already exists." % self.folder)

    def create_parser(self, name):
        '''
        :param name: string
        :return: parsed arguments
        '''

        p = argparse.ArgumentParser(
	        prog=name,
	        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	        description='Horseshoe variable selection with negative binomial regression model.'
        )

        p.add_argument(
	        '--seed',
	        help = 'simulation seed',
            default=0, type=int
        )

        p.add_argument(
	        '--flag', '--flag',
	        help = 'large or small simulation',
	        default=0, type=int
        )
        return p

    def main(self, opts):
        # prepare dataset
        self.real_data(opts.flag)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.X, self.Y, test_size=0.2, random_state=opts.seed)
        (n, p) = Xtrain.shape

        model = NegBinSSMCMC(opts.seed)
        model.set_data(Xtrain, Ytrain)
        model.set_prior(0.01, 0.01, 4, p)  # set prior on over-dispersion parameter r
        model.set_mcmc(10000, 3000, 1000)
        model.run_mcmc(True)
        time = model.get_time()

        gamma_mat = model.get_gamma_mat()
        beta_mat = model.get_beta_mat()
        beta0_vec = model.get_beta0_vec()
        r_vec = model.get_r_vec()

        gamma = np.median(gamma_mat, 0) > 0.5
        beta = np.median(beta_mat, 0)
        beta0 = np.median(beta0_vec)
        r = np.median(r_vec)
        m = sum(gamma)

        mu = np.mean(np.exp(np.matmul(Xtrain, beta_mat.transpose()) + beta0_vec + np.log(r_vec)), 1)
        kappa = np.mean(1.0 / r_vec)
        train_pearson = np.sum(np.power((Ytrain - mu) / np.sqrt(mu * (kappa * mu)), 2))
        print('seed {} train pearson residual is {}'.format(opts.seed, train_pearson))

        mu = np.mean(np.exp(np.matmul(Xtest, beta_mat.transpose()) + beta0_vec + np.log(r_vec)), 1)
        kappa = np.mean(1.0 / r_vec)
        test_pearson = np.sum(np.power((Ytest - mu) / np.sqrt(mu * (kappa * mu)), 2))
        print('seed {} test pearson residual is {}'.format(opts.seed, test_pearson))

        # prediction
        pred = np.exp(np.matmul(Xtrain, beta)+ beta0 + np.log(r))
        train_rmse = rmse(Ytrain, pred)
        print('seed {} train RMSE is {}'.format(opts.seed, train_rmse))

        pred = np.exp(np.matmul(Xtest, beta)+ beta0 + np.log(r))
        test_rmse = rmse(Ytest, pred)
        print('seed {} test RMSE is {}'.format(opts.seed, test_rmse))

        results = [opts.seed, train_rmse, test_rmse, train_pearson, test_pearson, m, time]

        neg_bin_simu = {"beta": beta, "gamma": gamma, "beta0": beta0, "r": r, "time": time,
                "train_rmse": train_rmse, "test_rmse": test_rmse, "train_pearson": train_pearson, "test_pearson": test_pearson,
                "m": m, "model": self.X.columns.get_values()[gamma],
                "beta_mat": beta_mat, "r_vec": r_vec, "beta0_vec": beta0_vec}

        pickle.dump(neg_bin_simu, open("%s//negbin_ss_mcmc_boston_%d.pickle" % (self.folder, opts.seed), "wb"), protocol=2)
        print(','.join(str(s) for s in results))


if __name__ == "__main__":
    app = App()
    app.run(sys.argv)