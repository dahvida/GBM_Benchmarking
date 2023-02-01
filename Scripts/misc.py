from fanova import fANOVA
import shap
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from itertools import chain, combinations
import matplotlib.pyplot as plt

###############################################################################

"""
MISCELLANEOUS FUNCTIONS FOR DRAWING FRAGMENTS AND RUNNING SHAPLEY / fANOVA ANALYSIS
"""

###############################################################################

"""
fANOVA FUNCTIONS

- unpack_trials:	get relevant values from Trials object from hyperopt
- powerset:		calculate indexes for all 1st order interactions and hyperparameter IDs
- augment_keys:	cuts powerset up to only 1st order interactions
- run_fANOVA:		executes fANOVA analysis given hyperopt optimization process
"""

def unpack_trials(trials):
    results = [x["loss"] for x in trials.results]
    results = np.array(results)
    values = list(trials.vals.values())
    keys = list(trials.vals.keys())
    matrix = np.empty((len(values[0]), len(values)))
    for i in range(len(values)):
        matrix[:,i] = values[i]
    
    return matrix, results, keys


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def augment_keys(keys):
    combinations = [x for x in powerset(keys)]
    n_combinations = np.sum(list(range(len(keys)+1)))
    combinations = combinations[1:n_combinations+1]
    for i in range(len(combinations)):
        combinations[i] = "*".join(combinations[i])
    return combinations    


def run_fANOVA(matrix, results):
    box = fANOVA(matrix, results)
    box.set_cutoffs(quantile=(0, 70))
    n_params = matrix.shape[1]
    array = np.empty((n_params,n_params))
    for i in range(n_params):
        for j in range(n_params):
            array[i,j] = box.quantify_importance((i,j))[(i,j)]["total importance"]
    
    unrolled = []
    for i in range(n_params):
        unrolled.append(array[i,i])
    for j in range(n_params-1):
        for k in range(j+1, n_params):
            unrolled.append(array[k,j])
   
    return [unrolled]


###############################################################################

"""
SHAPLEY FUNCTIONS

- get_importances:	get Shapley values from training set and GBM classifier 
			for input variables
- match_top_hits:	find overlapping bits between two ranked lists across all tasks
- compare_shaps:	given three sets of Shapley values, run all comparisons
"""

def get_importances(x, model):
    explainer = shap.Explainer(model)
    try:
        shap_values = explainer(x).values[:,:,1]
    except:
        shap_values = explainer(x).values
    shap_values = np.abs(shap_values)
    shap_values = np.mean(shap_values, axis=0)
    
    return shap_values
    

def match_top_hits(slice_1, slice_2):
    slice_1 = slice_1[-20:]
    slice_2 = slice_2[-20:]
    
    combined = np.concatenate((slice_1, slice_2), axis=0)
    uniques = np.unique(combined)
    
    return 1 - (len(uniques) / 20)
    

def compare_shaps(shap_1, shap_2, shap_3):
    comp_1_2 = [0]*shap_1.shape[1]
    comp_1_3 = [0]*shap_1.shape[1]
    comp_2_3 = [0]*shap_1.shape[1]
    
    for i in range(shap_1.shape[1]):
        slice_1 = np.argsort(shap_1[:,i])
        slice_2 = np.argsort(shap_2[:,i])
        slice_3 = np.argsort(shap_3[:,i])
        
        comp_1_2 = match_top_hits(slice_1, slice_2)
        comp_1_3 = match_top_hits(slice_1, slice_3)
        comp_2_3 = match_top_hits(slice_2, slice_3)
        
    return np.mean(comp_1_2), np.mean(comp_1_3), np.mean(comp_2_3)


###############################################################################

"""
FRAGMENT FUNCTIONS

- get_ecfp_info:	compute ECFPs with bit info
- get_shap:		get most important Shapley values given GBM and ECFPs
- process_shap:	create tuples with molecule, bit id and bit info
"""

def get_ecfp_info(mols):
    array = np.empty((len(mols), 1024), dtype=np.float32)
    info_box = []
    fps = [0]*len(mols)
    for i in range(len(mols)):
        info = {}
        fps[i] = AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, 1024, bitInfo=info)
        info_box.append(info)
        
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])    
    
    return array, info_box
    

def get_shap(model, ecfp):
    explainer = shap.Explainer(model)
    shaps = explainer(ecfp).values
    if len(shaps.shape) == 3:
        shaps = shaps[:,:,0]
    shaps = np.abs(shaps)
    shaps = np.mean(shaps, axis=0)
    top_shaps = np.argsort(shaps)[-20:]
    return np.array(top_shaps, dtype = np.int16)
    

def process_shap(ecfp,
                 mols,
                 shap_idx,
                 info):
    box = [0]*len(shap_idx)
    for i in range(len(shap_idx)):
        target_idx = shap_idx[i]
        mol_idx = np.where(ecfp[:, target_idx] == 1)[0][0]
        box[i] = (mols[mol_idx],
                  shap_idx[i],
                  info[mol_idx])
        
    return box
    










