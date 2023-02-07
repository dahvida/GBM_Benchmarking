from utils import eval_boosters, eval_dataset
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_iters', default="1")		#placeholder needed for the function, keep to 1
parser.add_argument('--opt_iters', default=500)		#number of optimization iterations
parser.add_argument('--fp_type', default="ECFP")    #molecular representation (ECFP, MACCS, RDKIT)
args = parser.parse_args()

###############################################################################

def main():
	#initialize parameters for the run and unpack args
	opt_iters = args.opt_iters
	run_iters = args.run_iters						#placeholder
	fp_type = args.fp_type
	sets = ["bace", "bbbp", "clintox", "tox21",
	 "muv", "hiv", "phos", "ntp", "oxi"]
	tasks = [1, 1, 2, 12, 17, 1, 5, 6, 10]
	repo = ["moleculenet"]*6 + ["moldata"]*3
	
	lgb_results = pd.DataFrame()
	
	#loop over all datasets
	for i in range(len(sets)):
	    	#acquire hyperparameter importances and first order interactions
	    	_, _, _, _, opts, keys, _ = eval_dataset(sets[i], 
                                             repo[i],
                                             tasks[i],
                                             fp_type,
                                             "lightgbm", 
                                             opt_iters=opt_iters,
                                             run_iters=run_iters,
                                             do_fanova=True,
                                             do_shap=False)
	    	#save importances in dataframe
	    	lgb_results[sets[i]] = opts[0][0]

	means = np.mean(np.array(lgb_results), axis=1)                     #average across all datasets
	lgb_results["Keys"] = keys                                         #save names of hyperparameters / 1st order interactions
	lgb_results["Means"] = means                                       #save means of hyperparameters / 1st order interactions
	lgb_results.sort_values("Means", ascending=False, inplace=True)    #sort
	lgb_results.to_csv("../Results/hyperparam_analysis.csv")           #store in .csv

if __name__ == "__main__":
	main()
