from utils import validate_booster
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

###############################################################################

def main():

	#initialize hyperparams for the analysis of moleculenet datasets
	opt_iters = 100
	run_iters = 5
	fp_type = "ECFP"
	dataset_names = ["HIV", "tox21", "MUV", "bace", "bbbp", "clintox"]
	task_n = [1, 12, 17, 1, 1, 2]
	
	#process all moleculenet datasets and save .csv files with metrics
	for i in range(len(dataset_names)):
    		validate_booster(
    			dataset_names[i],
                  	"moleculenet",
                  	task_n[i],
                  	fp_type,
                  	opt_iters,
                  	run_iters,
                  	GBM = "lightgbm")
        	
        #adapt parameters for moldata datasets	
	run_iters = 5
	dataset_names = ["phos", "ntp", "oxi"]
	task_n = [5, 6, 10]

	#process all moldata datasets and save .csv files with metrics
	for i in range(len(dataset_names)):
    		validate_booster(
    			dataset_names[i],
                  	"moldata",
                  	task_n[i],
                  	fp_type,
                  	opt_iters,
                  	run_iters,
                  	GBM = "lightgbm")
                
if __name__ == "__main__":
	main()   
