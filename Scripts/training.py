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
    if booster_type == "xgboost":
        index = ["max_leaves", "max_depth", "min_child_samples"]

    elif booster_type == "lightgbm":
        index = ["num_leaves", "max_depth", "min_child_samples", 
                         "subsample_freq"]

    elif booster_type == "catboost":
        index = ["depth", "leaf_estimation_iterations", "random_strength"]
    
    else:
        index = ["n_estimators", "max_depth"]
    
    for x in index:
        params[x] = int(params[x])
    
    return params
    

def train_model(x, y, x_val, y_val, params, booster_type):
    params = fix_params(params, booster_type)

    if booster_type == "xgboost":
        model = XGBClassifier(
            random_state =        np.random.randint(0, 1000),
            n_estimators =        1000,
            verbosity =           0,
            use_label_encoder =   False,
            **params)
        model.fit(x, y,
                  eval_set = [(x_val, y_val)],
                  early_stopping_rounds = 50,
                  verbose=False
                  )
        
    if booster_type == "lightgbm":
        early_stopping = lightgbm.early_stopping(stopping_rounds=50, verbose=False)
        model = LGBMClassifier(
            pos_subsample =       1,
            random_state =        np.random.randint(0, 1000),
            n_estimators =        1000,
            boosting_type =       "dart",
            max_bin =             15,
            verbose=             -10,   
            **params)
        model.fit(x, y, 
                  eval_set=(x_val, y_val), 
                  callbacks=[early_stopping])
        
    if booster_type == "catboost":
        model = CatBoostClassifier(
            random_seed =                 np.random.randint(0, 1000),
            iterations =                  1000,
            bootstrap_type =              "Bernoulli",
            use_best_model =              True,
            **params)
        model.fit(x, y,
                  eval_set = (x_val, y_val),
                  early_stopping_rounds=50,
                  verbose=False)
        
    if booster_type == "random_forest":
        model = RandomForestClassifier(n_estimators = params["n_estimators"],
                                       criterion = params["criterion"],
                                       max_depth = params["max_depth"],
                                       min_samples_split = params["min_samples_split"],
                                       min_samples_leaf = params["min_samples_leaf"],
                                       max_features = params["max_features"],
                                       class_weight = {0:1, 1:params["class_weight"]},
                                       n_jobs=-1)
        model.fit(x, y)
        
    return model

