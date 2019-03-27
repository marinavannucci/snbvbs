import time
import os


def job(seeds, flag, output_dir):
    for seed in seeds:
        os.system("nohup python -u py_one_ss_mcmc.py --seed {} --flag {} > ./{}/{}.out 2>&1 &".format(seed, flag, output_dir, seed))

if __name__ == "__main__":
    flag = 1
    if flag == 1:
        method = "%s" % ("ss_mcmc_boston_large")
    else:
        method = "%s" % ("ss_mcmc_boston")

    if not os.path.exists(method):
        os.makedirs(method)
        print("Directory %s is created." % method)
    else:
        print("Directory %s already exists." % method)

    seeds = list(range(0, 100))
    num_jobs = len(seeds)
    num_batch = 20
    job_batchs = [seeds[i: i+num_batch] for i in range(0, num_jobs, num_batch)]

    for job_batch in job_batchs:
        job(job_batch, flag, method)
        time.sleep(60 * 2)