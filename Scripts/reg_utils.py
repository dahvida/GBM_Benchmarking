from sklearn.metrics import *
import numpy as np
from loaders import *
from training import fix_params
import lightgbm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from hyperopt import tpe, hp, fmin, Trials
from misc import *
import time
import pandas as pd
import pickle as pkl
import json
import warnings
warnings.filterwarnings("ignore")

##################################################################################

"""
DATASET PROCESSING FUNCTIONS

- eval_dataset:	given a dataset and GBM algorithm it acquires all performance
			metrics and carries out Shapley / fANOVA analysis
			
- eval_boosters:	loops eval_dataset over all possible GBM algorithms for a given
			dataset and saves the formatted outputs in Results
			
- validate_booster:	compares shapley values overlap between two independent optimization
			runs for the same GBM algorithm
"""

#-----------------------------------------------------------------------------#

def create_reg_space(booster_type):
    
    #create hyperopt param space dict for XGBoost
    if booster_type == "xgboost":
        space = {
            'max_leaves':           hp.qloguniform('max_leaves', np.log(10), np.log(10000), 1),
            'learning_rate':        hp.loguniform('learning_rate', np.log(0.001), 0),
            'max_depth':            hp.quniform('max_depth', 3, 12, 1),
            
            'min_child_samples':    hp.qloguniform('min_child_samples', 0, 5, 1),
            "gamma":                hp.uniform("gamma", 0, 15),
            "min_child_weight":     hp.loguniform("min_child_weight", -5, 5),
            
            'reg_alpha':            hp.loguniform('reg_alpha', -5, 2),
            'reg_lambda':           hp.loguniform('reg_lambda', -5, 2),
            "max_delta_step":       hp.loguniform("max_delta_step", -5, 2),
            
            "colsample_bytree":     hp.uniform('colsample_bytree', 0.1, 1),
            "colsample_bylevel":    hp.uniform("colsample_bylevel", 0.1, 1),
            "subsample":            hp.uniform("subsample", 0.2, 1),
            
        }
        
    #create hyperopt param space dict for lightgbm
    if booster_type == "lightgbm":
        space = {
                'num_leaves':           hp.qloguniform('num_leaves', np.log(10), np.log(10000), 1),
                'learning_rate':        hp.loguniform('learning_rate', np.log(0.001), 0),
                'max_depth':            hp.quniform('max_depth', 3, 12, 1),
            
                'min_child_samples':    hp.qloguniform('min_child_samples', 0, 5, 1),
                "min_split_gain":       hp.uniform("min_split_gain", 0, 15),
                "min_child_weight":     hp.loguniform("min_child_weight", -5, 5),
            
                'reg_alpha':            hp.loguniform('reg_alpha', -5, 2),
                'reg_lambda':           hp.loguniform('reg_lambda', -5, 2),
            
                "colsample_bytree":     hp.uniform('colsample_bytree', 0.1, 1),
                "bagging_fraction":        hp.uniform("neg_subsample", 0.1, 1),
                "subsample_freq":       hp.quniform("subsample_freq", 0, 30, 1),
            
                }
        
    #create hyperopt param space dict for catboost
    if booster_type == "catboost":
        space = {
            'learning_rate':                hp.loguniform('learning_rate', np.log(0.001), 0),
            'depth':                        hp.quniform('depth', 3, 12, 1),
            "leaf_estimation_iterations":   hp.quniform("leaf_estimation_iterations", 1, 20, 1),
                        
            'l2_leaf_reg':                  hp.loguniform('l2_leaf_reg', -5, 2),
            
            "random_strength":              hp.quniform('random_strength', 1, 20, 1),
            "subsample":                    hp.uniform("subsample", 0.1, 1),
            "sampling_frequency":           hp.choice("sampling_frequency", ["PerTree", "PerTreeLevel"]),
            "colsample_bylevel":            hp.uniform("colsample_bylevel", 0.1, 1),
            "langevin":                     hp.choice("langevin", [True, False])
        }      
        
        
    return space

#-----------------------------------------------------------------------------#

def train_reg(x, y, x_val, y_val, params, booster_type):
    #fix param dict
    params = fix_params(params, booster_type)
    
    #train xgboost model
    if booster_type == "xgboost":
        model = XGBRegressor(
            random_state =        np.random.randint(0, 1000),   #force random seed
            n_estimators =        1000,
            verbosity =           0,
            use_label_encoder =   False,                        #prevents warning
            **params)
        model.fit(x, y,
                  eval_set = [(x_val, y_val)],
                  early_stopping_rounds = 50,                   #set iters for early stopping on val
                  verbose=False
                  )
    
    #train lightgbm model
    if booster_type == "lightgbm":
        #create early stopping callback
        early_stopping = lightgbm.early_stopping(stopping_rounds=50, verbose=False)
        
        model = LGBMRegressor(
            random_state =        np.random.randint(0, 1000),   #force random seed
            n_estimators =        1000,
            max_bin =             15,                           #set low n of bins for speed
            verbose=             -10,   
            **params)
        model.fit(x, y, 
                  eval_set=(x_val, y_val), 
                  callbacks=[early_stopping])
        
    #train catboost model
    if booster_type == "catboost":
        model = CatBoostRegressor(
            random_seed =                 np.random.randint(0, 1000),   #force random seed
            iterations =                  1000,                 
            bootstrap_type =              "Bernoulli",                  #set to Bernoulli for speed
            use_best_model =              True,
            **params)
        model.fit(x, y,
                  eval_set = (x_val, y_val),
                  early_stopping_rounds=50,                             #set iters for early stopping on val
                  verbose=False)
        
    return model

#-----------------------------------------------------------------------------#

def optimize_reg(
                 x_train, y_train,
                 x_val, y_val,
                 booster_type,
                 space,
                 iters = 100,
                 metric = mean_squared_error
                 ):
            
            #define eval function
            def model_eval(args):
                #train using moldata splits from input
                params = args
                model = train_reg(x_train, y_train, x_val, y_val, params, booster_type)
                
                #predict logits
                preds = model.predict(x_val)
                
                return np.sqrt(metric(y_val, preds))
            
            #create trials object
            trials = Trials()
            
            #run optimization
            optimum = fmin(
                fn = model_eval,
                space = space,
                algo = tpe.suggest,
                max_evals = iters,    
                trials = trials,
                verbose = False
                )
            
            #fix choice params from catboost
            if booster_type == "catboost":
                optimum["sampling_frequency"] = ["PerTree", "PerTreeLevel"][optimum["sampling_frequency"]]
                optimum["langevin"] = [True, False][optimum["langevin"]]
                
            return optimum


#-----------------------------------------------------------------------------#

def eval_reg(dataset_name,
             fp_type,
                 booster_type,
                 opt_iters = 100,
                 run_iters = 50,
                 do_shap = True):
    
    #prealloc results storage
    print("--Booster type:", booster_type)
    rmse = np.empty((run_iters))
    times = np.empty((run_iters))
    
    #prealloc shapley vectors
    if fp_type == "ECFP":
        shaps = np.zeros((1024))
    elif fp_type == "MACCS":
        shaps = np.zeros((167))
    elif fp_type == "RDKIT":
        shaps = np.zeros((208))
    

    #create appropriate param space
    space = create_reg_space(booster_type)
        
    train_x, train_y, test_x, test_y, val_x, val_y = chembl_loader(dataset_name, fp_type)
    opt = optimize_reg(train_x, train_y,
                           val_x, val_y,
                           booster_type, space, iters=opt_iters)
            
    print("Optimization finished")
        
    #loop over evaluation iters
    for j in range(run_iters):
            
            #train and monitor time
            start = time.time()
            model = train_reg(train_x, train_y, val_x, val_y, opt, booster_type)
            end = time.time()
            times[j] = end - start
            
            #get logits and metrics
            preds = model.predict(test_x)
            rmse[j] = np.sqrt(mean_squared_error(test_y, preds))
            
            #option to store shaps on each evaluation
            if do_shap is True:
                shaps[:] += get_importances(train_x, model)
        
        #average shaps across all evaluations
    shaps[:] = shaps[:] / run_iters
    print("Evaluation finished")
    
    return rmse, times, shaps
        
#-----------------------------------------------------------------------------#        
        
def eval_boosters_reg(
        dataset_name,
        fp_type,
        opt_iters = 100,
        run_iters = 50):
    
    prefix = "../Results/"
    print("Evaluation start for:", dataset_name)
    
    #get metrics for xgboost
    rmse_1, times_1, shaps_1 = eval_reg(dataset_name, 
                                        fp_type,
                                        "xgboost",
                                        opt_iters=opt_iters,
                                        run_iters=run_iters)
    
    #get metrics for lightgbm
    rmse_2, times_2, shaps_2 = eval_reg(dataset_name, 
                                        fp_type,
                                        "lightgbm",
                                        opt_iters=opt_iters,
                                        run_iters=run_iters)
    
    #get metrics for catboost
    rmse_3, times_3, shaps_3 = eval_reg(dataset_name, 
                                        fp_type,
                                        "catboost",
                                        opt_iters=opt_iters,
                                        run_iters=run_iters)
        
    #store classification&times in dataframe and save to .csv
    performance = pd.DataFrame({"RMSE - XGB": rmse_1,
                                "RMSE - LGB": rmse_2,
                                "RMSE - CB": rmse_3,
                                "T - XGB": times_1,
                                "T - LGB": times_2,
                                "T - CB": times_3
                                })
    performance.to_csv(prefix + dataset_name + "_performance.csv")
    
    shaps_1 = shaps_1.reshape(-1,1)
    shaps_2 = shaps_2.reshape(-1,1)
    shaps_3 = shaps_3.reshape(-1,1)
    
    #store shapley overlaps in dict and save to .txt
    comp_1_2, comp_1_3, comp_2_3 = compare_shaps(shaps_1, shaps_2, shaps_3)
    shap_comparison = {"XGB vs LGB": comp_1_2,
                       "XGB vs CB": comp_1_3,
                       "LGB vs CB": comp_2_3}
    with open(prefix + dataset_name + '_shap_comp.txt', 'w') as file:
    	file.write(json.dumps(shap_comparison))	
            
    print("Job finished")

#-----------------------------------------------------------------------------#

def validate_booster(dataset_name,
                  dataset_rep,
                  task_n,
                  fp_type,
                  opt_iters = 100,
                  run_iters = 50,
                  GBM = "lightGBM"):
    
    prefix = "../Results/"
    print("Evaluation start for:", dataset_name)
        
    #get shapleys from first optimization run
    _, _, _, _, _, _, shaps_1 = eval_reg(dataset_name, 
                                         fp_type,
                                         "lightgbm", 
                                         opt_iters=opt_iters,
                                         run_iters=run_iters)
    
    #get shapleys from second optimization run
    _, _, _, _, _, _, shaps_2 = eval_reg(dataset_name, 
                                         fp_type,
                                         "lightgbm", 
                                         opt_iters=opt_iters,
                                         run_iters=run_iters)
    
    #get overlaps, store in dict and then to .txt
    comp_1_2, comp_1_3, comp_2_3 = compare_shaps(shaps_1, shaps_2, shaps_1)
    shap_comparison = {"Run_1 vs Run_2": comp_1_2}
    with open(prefix + dataset_name + '_shap_comp.txt', 'w') as file:
    	file.write(json.dumps(shap_comparison))	
            
    print("Job finished")

#-----------------------------------------------------------------------------#



