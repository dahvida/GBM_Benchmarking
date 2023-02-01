from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import numpy as np
from loaders import *
from training import *
from optimization import *
from misc import *
import time
import pandas as pd
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

##################################################################################

"""
DATASET PROCESSING FUNCTIONS

- eval_dataset:	given a dataset and GBM algorithm it acquires all performance
			metrics and carries out Shapley / fANOVA analysis
			
- eval_boosters:	loops eval_dataset over all possible GBM algorithms for a given
			dataset and saves the formatted outputs in Results
"""

def eval_dataset(dataset_name,
                 dataset_source,
                 task_n,
                 fp_type,
                 booster_type,
                 opt_iters = 50,
                 run_iters = 50,
                 do_fanova = True,
                 do_shap = True):
    
    print("--Booster type:", booster_type)
    pr_auc = np.empty((run_iters, task_n))
    roc_auc = np.empty((run_iters, task_n))
    
    times = np.empty((run_iters, task_n))
    opts = [0]*task_n
    if fp_type == "ECFP":
        shaps = np.empty((1024, task_n))
    elif fp_type == "MACCS":
        shaps = np.empty((167, task_n))
    elif fp_type == "RDKIT":
        shaps = np.empty((208, task_n))
    keys = []
    
    for i in range(task_n):
        print("--Task ID:", i)
        space = create_param_space(booster_type)
        if dataset_source == "moleculenet":
            x, y = molnet_loader(dataset_name, i, fp_type)
            opt, trials = optimize_moleculenet(x, y, booster_type, space, iters=opt_iters)
        else:
            train_x, train_y, test_x, test_y, val_x, val_y = moldata_loader(dataset_name, i, fp_type)
            opt, trials = optimize_moldata(train_x, train_y,
                                            val_x, val_y,
                                            booster_type, space, iters=opt_iters)
            
        
        print("Optimization finished")
        
        if do_fanova is True:
            matrix, results, keys = unpack_trials(trials)
            opts[i] = run_fANOVA(matrix, results)
        
        for j in range(run_iters):
            
            if fp_type == "ECFP":
                s_temp = np.zeros((1024,))
            elif fp_type == "MACCS":
                s_temp = np.zeros((167,))
                
            if dataset_source == "moleculenet":
                train_x, val_x, train_y, val_y = train_test_split(x, y,
                                                  stratify=y,
                                                  test_size=0.2)                
                test_x, val_x, test_y, val_y = train_test_split(val_x, val_y,
                                                stratify=val_y,
                                                test_size=0.5) 
            else:
                pass
            
            start = time.time()
            model = train_model(train_x, train_y, val_x, val_y, opt, booster_type)
            end = time.time()
            times[j, i] = end - start

            preds = model.predict_proba(test_x)[:,1]
            
            pr_auc[j, i] = average_precision_score(test_y, preds)
            roc_auc[j, i] = roc_auc_score(test_y, preds)
            
            if do_shap is True:
                shaps[:, i] += get_importances(train_x, model)
        
        shaps[:, i] = shaps[:, i] / run_iters
        print("Evaluation finished")
    
    pr_auc = np.mean(pr_auc, axis=1)
    roc_auc = np.mean(roc_auc, axis=1)
    times = np.mean(times, axis=1)
    
    return pr_auc, roc_auc, model, times, opts, augment_keys(keys), shaps
        
        
def eval_boosters(dataset_name,
                  dataset_rep,
                  task_n,
                  fp_type,
                  opt_iters = 100,
                  run_iters = 50):
    
    prefix = "../Results/"
    print("Evaluation start for:", dataset_name)
    
    pr_auc_1, roc_auc_1, model_1, times_1, opts_1, keys_1, shaps_1 = eval_dataset(dataset_name, 
                                                                  dataset_rep,
                                                                  task_n,
                                                                  fp_type,
                                                                  "xgboost", 
                                                                  opt_iters=opt_iters,
                                                                  run_iters=run_iters)

    pr_auc_2, roc_auc_2, model_2, times_2, opts_2, keys_2, shaps_2 = eval_dataset(dataset_name, 
                                                                  dataset_rep,
                                                                  task_n,
                                                                  fp_type,
                                                                  "lightgbm", 
                                                                  opt_iters=opt_iters,
                                                                  run_iters=run_iters)

    pr_auc_3, roc_auc_3, model_3, times_3, opts_3, keys_3, shaps_3 = eval_dataset(dataset_name, 
                                                                  dataset_rep,
                                                                  task_n,
                                                                  fp_type,
                                                                  "catboost", 
                                                                  opt_iters=opt_iters,
                                                                  run_iters=run_iters)
        
    performance = pd.DataFrame({"PR - XGB": pr_auc_1,
                                "PR - LGB": pr_auc_2,
                                "PR - CB": pr_auc_3,
                                "ROC - XGB": roc_auc_1,
                                "ROC - LGB": roc_auc_2,
                                "ROC - CB": roc_auc_3,
                                "T - XGB": times_1,
                                "T - LGB": times_2,
                                "T - CB": times_3
                                })
    performance.to_csv(prefix + dataset_name + "_performance.csv")

    xgb_dict = dict(zip(keys_1, opts_1))
    lgb_dict = dict(zip(keys_2, opts_2))
    cb_dict = dict(zip(keys_3, opts_3))

    comp_1_2, comp_1_3, comp_2_3 = compare_shaps(shaps_1, shaps_2, shaps_3)

    shap_comparison = {"XGB vs LGB": comp_1_2,
                       "XGB vs CB": comp_1_3,
                       "LGB vs CB": comp_2_3}

    summary = [xgb_dict, lgb_dict, cb_dict, shap_comparison]
    with open(prefix + dataset_name + "_summary.pkl", "wb") as output_file:
            pkl.dump(summary, output_file)
            
    print("Job finished")




