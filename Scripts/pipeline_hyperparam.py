from utils import eval_boosters, eval_dataset
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
import argparse

###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--repo', default="all")			#repository to use for the analysis (moleculenet, moldata, all)
parser.add_argument('--opt_iters', default=500, type=int)	#number of optimization iterations
parser.add_argument('--fp_type', default="ECFP")    		#molecular representation (ECFP, MACCS, RDKIT)
args = parser.parse_args()

###############################################################################

def main(
	opt_iters,
	repo,
	fp_type
	):
	
	#initialize pandas dataframe for the run
	lgb_results = pd.DataFrame()
	
	#initialize hyperparams for the analysis of the datasets
	if repo == "all":
		dataset_names = [
			"HIV", "tox21", "MUV", "bace", "bbbp", "clintox",
			"phos", "ntp", "oxi"
			]
		task_n = [1, 12, 17, 1, 1, 2, 5, 6, 10]
		repository_names = ["moleculenet"]*6 + ["moldata"]*3
		iters = [iters_moleculenet]*6 + [iters_moldata]*3
		
	if repo == "moleculenet":
		dataset_names = ["HIV", "tox21", "MUV", "bace", "bbbp", "clintox"]
		task_n = [1, 12, 17, 1, 1, 2]		 ]
		repository_names = ["moleculenet"]*6
		iters = [iters_moleculenet]*6
		
	if repo == "moldata":
		dataset_names = ["phos", "ntp", "oxi"]
		task_n = [5, 6, 10]
		repository_names = ["moldata"]*3
		iters = [iters_moldata]*3	
	
	#loop over all datasets
	for i in range(len(sets)):
	    	#acquire hyperparameter importances and first order interactions
	    	_, _, _, _, opts, keys, _ = eval_dataset(
                                             dataset_names[i], 
                                             repo[i],
                                             tasks[i],
                                             fp_type,
                                             "lightgbm", 
                                             opt_iters=opt_iters,
                                             run_iters=1,
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
	main(
	opt_iters = args.opt_iters,
	repo = args.opt_repo,
	fp_type = args.fp_type
	)   
