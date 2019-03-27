import glob
import pickle
import csv
import os
import numpy as np
import pandas as pd
import re

result = ["seed", "train_rmse", "test_rmse", "train_pearson", "test_pearson", "m", "time"]
method = "hs_mcmc_boston"

csvfile = "%s//%s.csv" % (method, method)
for filepath in glob.glob("%s\\*.pickle" % method):
    with open(csvfile, "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(result)
    
        dict = pickle.load(open(filepath, "rb"))

    result = [int(re.findall('\d+', filepath)[0]), dict["train_rmse"], dict["test_rmse"], 
              dict["train_pearson"], dict["test_pearson"], dict["m"], dict["time"]]
 
with open(csvfile, "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(result)

