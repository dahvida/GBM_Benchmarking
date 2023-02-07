import lightgbm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#####################################################################################

"""
TRAINING FUNCTIONS

- fix_params: 	adjusts values from hyperopt before feeding in model
- train_model:	trains a GBM classifier given splits, params and type of algorithm
"""

def fix_params(params, booster_type):
    
    #get params to convert to int for XGBoost
    if booster_type == "xgboost":
        index = ["max_leaves", "max_depth", "min_child_samples"]

    #get params to convert to int for LightGBM
    elif booster_type == "lightgbm":
        index = ["num_leaves", "max_depth", "min_child_samples", 
                         "subsample_freq"]

    #get params to convert to int for CatBoost
    elif booster_type == "catboost":
        index = ["depth", "leaf_estimation_iterations", "random_strength"]
    
    #convert params to int
    for x in index:
        params[x] = int(params[x])
    
    return params
    

def train_model(x, y, x_val, y_val, params, booster_type):
    #fix param dict
    params = fix_params(params, booster_type)
    
    #train xgboost model
    if booster_type == "xgboost":
        model = XGBClassifier(
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
        
        model = LGBMClassifier(
            pos_subsample =       1,                            #set to enable majority class sampling
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
        model = CatBoostClassifier(
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

