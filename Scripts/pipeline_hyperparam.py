from utils import eval_boosters, eval_dataset
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

#initialize parameters for the run
opt_iters = 500
run_iters = 1						#placeholder
sets = ["bace", "bbbp", "clintox", "tox21",
	 "muv", "hiv", "phos", "ntp", "oxi"]
tasks = [1, 1, 2, 12, 17, 1, 5, 6, 10]
repo = ["moleculenet"]*6 + ["moldata"]*3
fp_type = "ECFP"
lgb_results = pd.DataFrame()

###############################################################################

def main():
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

	means = np.mean(np.array(lgb_results), axis=1)			#average across all datasets
	lgb_results["Keys"] = keys						#save names of hyperparameters / 1st order interactions
	lgb_results["Means"] = means						#save means of hyperparameters / 1st order interactions
	lgb_results.sort_values("Means", ascending=False, inplace=True)	#sort
	lgb_results.to_csv("../Results/hyperparam_analysis.csv")		#store in .csv

if __name__ == "main":
	main()
