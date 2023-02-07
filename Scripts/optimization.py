from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from hyperopt import tpe, hp, fmin, Trials
import numpy as np
from training import *

###############################################################################################

"""
OPTIMIZATION FUNCTIONS

- create_param_space:		creates hyperparam space depending on GBM type
- optimize_moldata:		optimizes hyperparams for given GBM on given task from moldata dataset
- optimize_moleculenet:	optimizes hyperparams for given GBM on given task from moleculenet dataset
"""

def create_param_space(booster_type):
    
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
            
            "scale_pos_weight":     hp.loguniform("scale_pos_weight", 0, 6)
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
                "neg_subsample":        hp.uniform("neg_subsample", 0.1, 1),
                "subsample_freq":       hp.quniform("subsample_freq", 0, 30, 1),
            
                "scale_pos_weight":     hp.loguniform("scale_pos_weight", 0, 6)
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
            "scale_pos_weight":             hp.loguniform("scale_pos_weight", 0, 6),
            "langevin":                     hp.choice("langevin", [True, False])
        }      
        
        
    return space


def optimize_moldata(
                 x_train, y_train,
                 x_val, y_val,
                 booster_type,
                 space,
                 iters = 100,
                 metric = average_precision_score
                 ):
            
            #define eval function
            def model_eval(args):
                #train using moldata splits from input
                params = args
                model = train_model(x_train, y_train, x_val, y_val, params, booster_type)
                
                #predict logits
                preds = model.predict_proba(x_val)[:,1]
                
                return 1-metric(y_val, preds)
            
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
                
            return optimum, trials


def optimize_moleculenet(
          x, y,
          booster_type,
          space,
          iters = 100,
          splits = 3,
          metric = average_precision_score
        ):
        
        #define eval function
        def model_eval(args):
                #read args and create box to store results
                params = args
                performance_box = []
                
                #loop over inner split routine
                for i in range(splits):
                    #create train/test/val according to Jiang et al
                    x1, x2, y1, y2 = train_test_split(x, y, stratify=y, test_size=0.2)    
                    x2, x3, y2, y3 = train_test_split(x2, y2, stratify=y2, test_size=0.5)
                    
                    #train on splits, predict logits and store performance
                    model = train_model(x1, y1, x2, y2, params, booster_type)
                    preds = model.predict_proba(x3)[:,1]
                    performance_box.append(metric(y3, preds))
            
                return 1-np.mean(performance_box)
        
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
                
        return optimum, trials
