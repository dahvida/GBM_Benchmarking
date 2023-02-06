from utils import eval_boosters, eval_dataset
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--opt_iters', default=100)		#number of optimization iterations
parser.add_argument('--iters_moleculenet', default=50)	#number of eval iterations for moleculenet datasets
parser.add_argument('--iters_moldata', default=5)		#number of eval iterations for moldata datasets
parser.add_argument('--fp_type', default="ECFP")		#molecular representation (ECFP, MACCS, RDKIT)
args = parser.parse_args()

###############################################################################

def main():
	#unpack from command line
	opt_iters = args.opt_iters
	iters_moleculenet = args.iters_moleculenet
	iters_moldata = args.iters_moldata
	fp_type = args.fp_type

	#initialize hyperparams for the analysis of the datasets
	dataset_names = [
			"HIV", "tox21", "MUV", "bace", "bbbp", "clintox",
			"phos", "ntp", "oxi"
			]
	task_n = [1, 12, 17, 1, 1, 2, 5, 6, 10]
	repository_names = ["moleculenet"]*9 + ["moldata"]*3
	iters = [iters_moleculenet]*9 + [iters_moldata]*3
	
	#process all moleculenet datasets and save .csv files with metrics
	for i in range(len(dataset_names)):
    		eval_boosters(
    			dataset_names[i],
                  	repository_names[i],
                  	task_n[i],
                  	fp_type,
                  	opt_iters,
                  	iters[i])

               
if __name__ == "__main__":
	main()                                   
                  
                  
                  
