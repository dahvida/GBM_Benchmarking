from utils_grid import eval_dataset
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

###############################################################################

def main():
	#initialize hyperparams for the analysis
    fp_types = ["ECFP", "MACCS", "RDKIT"]
    results_sider = pd.DataFrame()
    results_fungal = pd.DataFrame()
	
    #loop over fp_types
    for i in range(3):
		#evaluate SIDER dataset with different grids
    		pr_100d, roc_100d = eval_lightgbm("sider",
                             "moleculenet",
                             27,
                             fp_types[i],
                             "default",
                             100,
                             50)
    		pr_30d, roc_30d = eval_lightgbm("sider",
                             "moleculenet",
                             27,
                             fp_types[i],
                             "default",
                             30,
                             50)
    		pr_30o, roc_30o = eval_lightgbm("sider",
                            "moleculenet",
                             27,
                             fp_types[i],
                             "optimal",
                             30,
                             50)
    		
    		#store result in dataframe
    		results_sider[fp_types[i]+"ROC_100_d"] = roc_100d
    		results_sider[fp_types[i]+"PR_100_d"] = pr_100d
    		results_sider[fp_types[i]+"ROC_30_d"] = roc_30d
    		results_sider[fp_types[i]+"PR_30_d"] = pr_30d
    		results_sider[fp_types[i]+"ROC_30_o"] = roc_30o
    		results_sider[fp_types[i]+"PR_30_o"] = pr_30o
    
    #save to .csv
    results_sider.to_csv("../Results/sider_grid.csv")
    
    #loop over fp_types
    for i in range(3):
    		#evaluate Trans dataset with different grids
    		pr_100d, roc_100d = eval_lightgbm("fungal",
                             "moldata",
                             9,
                             fp_types[i],
                             "default",
                             100,
                             5)
    		pr_30d, roc_30d = eval_lightgbm("fungal",
                             "moldata",
                             9,
                             fp_types[i],
                             "default",
                             30,
                             5)
    		pr_30o, roc_30o = eval_lightgbm("fungal",
                             "moldata",
                             9,
                             fp_types[i],
                             "optimal",
                             30,
                             5)
    
    		#store result in dataframe
    		results_fungal[fp_types[i]+"_ROC_100_d"] = roc_100d
    		results_fungal[fp_types[i]+"_PR_100_d"] = pr_100d
    		results_fungal[fp_types[i]+"_ROC_30_d"] = roc_30d
    		results_fungal[fp_types[i]+"_PR_30_d"] = pr_30d
    		results_fungal[fp_types[i]+"_ROC_30_o"] = roc_30o
    		results_fungal[fp_types[i]+"_PR_30_o"] = pr_30o
    
    #save to .csv
    results_fungal.to_csv("../Results/fungal_grid.csv")

if __name__ == "__main__":
	main()


		
