from optimization import *
from training import *
from loaders import *
from misc import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from rdkit.Chem.Draw import IPythonConsole
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="BACE")		#Dataset to use for the analysis
parser.add_argument('--repo', default="moleculenet")		#Repository of the dataset (moleculenet, moldata)
parser.add_argument('--task_n', default="1")			#number of tasks for the chosen dataset
parser.add_argument('--opt_iters', default=100)		#number of optimization iterations
parser.add_argument('--fp_type', default="ECFP")		#molecular representation (ECFP, MACCS, RDKIT)
args = parser.parse_args()

###############################################################################

def main():
    #initialize hyperparams for the analysis and unpack args
    opt_iters = args.opt_iters
    dataset = args.dataset
    source = args.repo
    task_n = args.task_n
    fp_type = args.fp_type
    boosters = ["xgboost", "lightgbm", "catboost"]
    spaces = [create_param_space(x) for x in boosters]
    models = [0]*3
    
	#load molecules and create train/val splits
    mols, y = molnet_loader(dataset, 0, None)
    ecfp, info = get_ecfp_info(mols)
    xt, xv, yt, yv = train_test_split(ecfp, y, test_size=0.1)
	
    #optimize each model over splits and save
    for i in range(len(models)):
    		opt, _ = optimize_moleculenet(ecfp,
                                      y,
                                      boosters[i],
                                      spaces[i],
                                      opt_iters)
    		models[i] = train_model(xt, yt, xv, yv, opt, boosters[i])

    #get top 20 most important variables for each model
    top_global_shaps = np.zeros((3, 20), dtype = np.int16)
    for i in range(len(models)):
	    top_global_shaps[i,:] = get_shap(models[i], ecfp)
	
    #save molecules containing relevant bits in list
    tpls_box = [0]*3
    for i in range(len(models)):
	    tpls_box[i] = process_shap(ecfp,
                         mols,
                         top_global_shaps[i, :],
                         info)

    #for each model, sort the molecules in correct order for drawing and create legend
    #list should be sorted such as [xgboost1, lightgbm1, catboost1, xgboost2, lightgbm2, ...]
    tpls = []
    legend = []
    for i in range(20):
	    for j in range(len(models)):
	        legend.append(boosters[j] + " - " + str(i + 1))
	        tpls.append(tpls_box[j][i])

    #draw and save fragments
    fig = Draw.DrawMorganBits(tpls,
                          molsPerRow=3, 
                          legends = legend,
                          subImgSize = (400, 400),
                          useSVG=False
                          )
    fig.save("../Results/all_fragments.png")

if __name__ == "__main__":
	main()














