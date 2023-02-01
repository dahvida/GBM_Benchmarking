from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin, Trials
import lightgbm as lgb
import numpy as np
from loaders import *
import warnings
warnings.filterwarnings("ignore")

###############################################################################

"""
DATASET PROCESSING FUCTIONS FOR HYPERPARAMETER GRID ANALYSIS WITH LIGHTGBM

- create_param_space:		creates hyperparam space depending on chosen grid type
- fix_params:			adjusts values from hyperopt before feeding in model
- train_model:			trains a LightGBM classifier given splits, params and grid type
- optimize_lightgbm:		optimizes a LightGBM classifier for given dataset
- eval_lightgbm:		acquires ROC-AUC and PR-AUC for a given dataset using LightGBM
"""

def create_param_space(grid_type = "default"):
    if grid_type == "default":
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
                "neg_subsample":        hp.uniform("neg_subsample", 0.1, 1),
                "subsample_freq":       hp.quniform("subsample_freq", 0, 30, 1),
                "scale_pos_weight":     hp.loguniform("scale_pos_weight", 0, 6)
                }
    else:
        space = {
                'learning_rate':        hp.loguniform('learning_rate', np.log(0.001), 0),
                "min_split_gain":       hp.uniform("min_split_gain", 0, 15),
                "min_child_weight":     hp.loguniform("min_child_weight", -5, 5),
                'reg_lambda':           hp.loguniform('reg_lambda', -5, 2),
                "neg_subsample":        hp.uniform("neg_subsample", 0.1, 1),
                "scale_pos_weight":     hp.loguniform("scale_pos_weight", 0, 6),
                "subsample_freq":       hp.quniform("subsample_freq", 0, 30, 1),
                }
        
    return space
    

def fix_params(params, grid_type):
    if grid_type == "default":
        index = ["num_leaves", "max_depth", "min_child_samples", "subsample_freq"]
    else:
        index = ["subsample_freq"]
    
    for x in index:
        params[x] = int(params[x])
    
    return params
    
        
def train_model(x, y, x_val, y_val, params, grid_type):
    params = fix_params(params, grid_type)
    early_stopping = lgb.early_stopping(stopping_rounds=50, verbose=False)
    model = lgb.LGBMClassifier(
            pos_subsample =       1,
            random_state =        np.random.randint(0, 1000),
            n_estimators =        1000,
            max_bin =             15,
            verbose=             -10,   
            **params)
    model.fit(x, y, 
                  eval_set=(x_val, y_val), 
                  callbacks=[early_stopping])
               
    return model
    

def optimize_lightgbm(
                 x_train = None,
                 y_train = None,
                 x_val = None,
                 y_val = None,
                 grid_type = "default",
                 space = None,
                 iters = 100,
                 splits = 3,
                 source = "moleculenet",
                 metric = average_precision_score
                 ):
            
            if source == "moldata":    
                def model_eval(args):
            
                    #define training loop
                    params = args
                    model = train_model(x_train, y_train, x_val, y_val, params, grid_type)
                    preds = model.predict_proba(x_val)[:,1]
                    return 1-metric(y_val, preds)
            
            else:
                def model_eval(args):
                    
                        params = args
                        performance_box = []
                        for i in range(splits):
                            x1, x2, y1, y2 = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)    
                            x2, x3, y2, y3 = train_test_split(x2, y2, stratify=y2, test_size=0.5)
                            model = train_model(x1, y1, x2, y2, params, grid_type)
                            preds = model.predict_proba(x3)[:,1]
                            performance_box.append(metric(y3, preds))
                    
                        return 1-np.mean(performance_box)
            
            trials = Trials()
            #get optimum hyperparameters
            optimum = fmin(
                fn = model_eval,
                space = space,
                algo = tpe.suggest,
                max_evals = iters,    
                trials = trials,
                verbose = False
                )
                
            return optimum


def eval_lightgbm(dataset_name,
                 dataset_source,
                 task_n,
                 fp_type,
                 grid_type,
                 opt_iters = 100,
                 run_iters = 50):
    
    pr_auc = np.empty((run_iters, task_n))
    roc_auc = np.empty((run_iters, task_n))
  
    
    for i in range(task_n):
        print("--Task ID:", i)
        space = create_param_space(grid_type)
        if dataset_source == "moleculenet":
            x, y = molnet_loader(dataset_name, i, fp_type)
            opt  = optimize_lightgbm(x_train = x,
                                     y_train = y,
                                     grid_type = grid_type,
                                     space = space,
                                     iters=opt_iters,
                                     source = dataset_source)
        else:
            train_x, train_y, test_x, test_y, val_x, val_y = moldata_loader(dataset_name, i, fp_type)
            opt = optimize_lightgbm(x_train = train_x,
                                    y_train = train_y,
                                    x_val = val_x,
                                    y_val = val_y,
                                    grid_type = grid_type,
                                    space = space,
                                    iters=opt_iters,
                                    source = dataset_source)
        
        
        for j in range(run_iters):
                           
            if dataset_source == "moleculenet":
                train_x, val_x, train_y, val_y = train_test_split(x, y,
                                                  stratify=y,
                                                  test_size=0.2)                
                test_x, val_x, test_y, val_y = train_test_split(val_x, val_y,
                                                stratify=val_y,
                                                test_size=0.5) 
            else:
                pass
            
            model = train_model(train_x, train_y, val_x, val_y, opt, grid_type)
            preds = model.predict_proba(test_x)[:,1]
            
            pr_auc[j, i] = average_precision_score(test_y, preds)
            roc_auc[j, i] = roc_auc_score(test_y, preds)
            
    pr_auc = np.mean(pr_auc, axis=1)
    roc_auc = np.mean(roc_auc, axis=1)
    
    return pr_auc, roc_auc
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
