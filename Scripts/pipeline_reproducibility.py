from utils import validate_booster
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
import argparse

###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--opt_iters', default=100, type=int)	        #number of optimization iterations
parser.add_argument('--repo', default="all")				#repository to use for the analysis (moleculenet, moldata, all)
parser.add_argument('--iters_moleculenet', default=50, type=int)	#number of eval iterations for moleculenet datasets
parser.add_argument('--iters_moldata', default=5, type=int)		#number of eval iterations for moldata datasets
parser.add_argument('--fp_type', default="ECFP")			#molecular representation (ECFP, MACCS, RDKIT)
parser.add_argument('--GBM', default="lightgbm")			#GBM to use for the script (lightgbm, xgboost, catboost)
args = parser.parse_args()

###############################################################################

def main(
	opt_iters,
	repo,
	iters_moleculenet,
	iters_moldata,
	fp_type,
	GBM
	):
	
	#initialize hyperparams for the analysis of the datasets
	if repo == "all":
		dataset_names = [
			"HIV", "tox21", "MUV", "bace", "bbbp", "clintox",
			"phos", "ntp", "oxi", "ache", "herg", "erbb1", "jak2",
			"cox2"
			]
		task_n = [1, 12, 17, 1, 1, 2, 5, 6, 10, 1, 1, 1, 1, 1]
		repository_names = ["moleculenet"]*6 + ["moldata"]*3 + ["chembl"]*5
		iters = [iters_moleculenet]*6 + [iters_moldata]*3 + [iters_chembl]*5
		
	elif repo == "moleculenet":
		dataset_names = ["HIV", "tox21", "MUV", "bace", "bbbp", "clintox"]
		task_n = [1, 12, 17, 1, 1, 2]
		repository_names = ["moleculenet"]*6
		iters = [iters_moleculenet]*6
		
	elif repo == "moldata":
		dataset_names = ["phos", "ntp", "oxi"]
		task_n = [5, 6, 10]
		repository_names = ["moldata"]*3
		iters = [iters_moldata]*3
        
	elif repo == "chembl":
		dataset_names = ["ache", "herg", "erbb1", "jak2", "cox2"]
		repository_names = ["chembl"]*5
		iters = [iters_moleculenet]*5
		task_n = [1,1,1,1,1]
	
	#process all moleculenet datasets and save .csv files with metrics
	for i in range(len(dataset_names)):
		validate_booster(
					dataset_names[i],
                  	repository_names[i],
                  	task_n[i],
                  	fp_type,
                  	opt_iters,
                  	iters[i],
                  	GBM = GBM)
        	                
if __name__ == "__main__":
	main(
	opt_iters = args.opt_iters,
	repo = args.repo,
	iters_moleculenet = args.iters_moleculenet,
	iters_moldata = args.iters_moldata,
	fp_type = args.fp_type,
	GBM = args.GBM
	)   
	
	
	
