from utils import eval_boosters, eval_dataset
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

#initialize hyperparams for the analysis of moleculenet datasets
opt_iters = 100
run_iters = 50
fp_type = "ECFP"
dataset_names = ["HIV", "tox21", "MUV", "bace", "bbbp", "clintox"]
task_n = [1, 12, 17, 1, 1, 2]

###############################################################################

def main():
	#process all moleculenet datasets and save .csv files with metrics
	for i in range(len(dataset_names)):
    		eval_boosters(dataset_names[i],
                  "moleculenet",
                  task_n[i],
                  fp_type,
                  opt_iters,
                  run_iters)

	#adapt parameters for moldata datasets
	run_iters = 5
	dataset_names = ["phos", "ntp", "oxi"]
	task_n = [5, 6, 10]

	#process all moldata datasets and save .csv files with metrics
	for i in range(len(dataset_names)):
    		eval_boosters(dataset_names[i],
                  "moldata",
                  task_n[i],
                  fp_type,
                  opt_iters,
                  run_iters)
                  
if __name__ = "main":
	main()                                   
                  
                  
                  
