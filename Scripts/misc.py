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
- powerset:		    calculate indexes for all 1st order interactions and hyperparameter IDs
- augment_keys:	    cuts powerset up to only 1st order interactions
- run_fANOVA:		executes fANOVA analysis given hyperopt optimization process
"""

#-----------------------------------------------------------------------------#

def unpack_trials(trials):
    #get loss values from optimization process
    results = [x["loss"] for x in trials.results]
    results = np.array(results)
    
    #get hyperparameter vals and names for each eval
    values = list(trials.vals.values())
    keys = list(trials.vals.keys())
    
    #store hyperparam combinations in matrix for fANOVA
    matrix = np.empty((len(values[0]), len(values)))
    for i in range(len(values)):
        matrix[:,i] = values[i]
    
    return matrix, results, keys

#-----------------------------------------------------------------------------#

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

#-----------------------------------------------------------------------------#

def augment_keys(keys):
    #get all possible interactions between hyperparam indexes
    combinations = [x for x in powerset(keys)]
    
    #slice combinations up to individual and 1st order interactions
    n_combinations = np.sum(list(range(len(keys)+1)))
    combinations = combinations[1:n_combinations+1]
    
    #create list with all possible hyperparam ID combinations
    for i in range(len(combinations)):
        combinations[i] = "*".join(combinations[i])
        
    return combinations    
    
#-----------------------------------------------------------------------------#

def run_fANOVA(matrix, results):
    #initialize fANOVA and remove worst 30% runs to focus eval on near-optimum
    box = fANOVA(matrix, results)
    box.set_cutoffs(quantile=(0, 70))
    
    #instantiate empty matrix with right shape
    n_params = matrix.shape[1]
    array = np.empty((n_params,n_params))
    
    #store results in matrix
    for i in range(n_params):
        for j in range(n_params):
            array[i,j] = box.quantify_importance((i,j))[(i,j)]["total importance"]
    
    #unroll matrix to match the output of augment_keys function
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
                    variables
- match_top_hits:	find overlapping bits between two ranked lists across all tasks
- compare_shaps:	given three sets of Shapley values, run all comparisons
"""

#-----------------------------------------------------------------------------#

def get_importances(x, model):
    #create explainer object for Shapley analysis
    explainer = shap.Explainer(model)
    
    #depending on GBM, get shap_values
    try:
        shap_values = explainer(x).values[:,:,1]
    except:
        shap_values = explainer(x).values
        
    #get global explanations
    shap_values = np.abs(shap_values)
    shap_values = np.mean(shap_values, axis=0)
    
    return shap_values
    
#-----------------------------------------------------------------------------#

def match_top_hits(slice_1, slice_2):
    #select top 20 vars
    slice_1 = slice_1[-20:]
    slice_2 = slice_2[-20:]
    
    #get number of unique var IDs between the two lists
    combined = np.concatenate((slice_1, slice_2), axis=0)
    uniques = np.unique(combined)
    
    #get number of variables that are present in only one list
    n_uniques = len(uniques) - 20
    
    return 1 - (n_uniques / 20)
    
#-----------------------------------------------------------------------------#    

def compare_shaps(shap_1, shap_2, shap_3):
    #preallocate list of right size depending on n_tasks
    comp_1_2 = [0]*shap_1.shape[1]
    comp_1_3 = [0]*shap_1.shape[1]
    comp_2_3 = [0]*shap_1.shape[1]
    
    #iterate on all tasks
    for i in range(shap_1.shape[1]):
        #sort var IDs according to Shapley values
        slice_1 = np.argsort(shap_1[:,i])
        slice_2 = np.argsort(shap_2[:,i])
        slice_3 = np.argsort(shap_3[:,i])
        
        #get Shapley overlaps for all combinations
        comp_1_2 = match_top_hits(slice_1, slice_2)
        comp_1_3 = match_top_hits(slice_1, slice_3)
        comp_2_3 = match_top_hits(slice_2, slice_3)
        
    return np.mean(comp_1_2), np.mean(comp_1_3), np.mean(comp_2_3)


###############################################################################

"""
FRAGMENT FUNCTIONS

- get_ecfp_info:	compute ECFPs with bit info
- get_shap:		    get most important Shapley values given GBM and ECFPs
- process_shap:	    create tuples with molecule, bit id and bit info
"""

#-----------------------------------------------------------------------------#

def get_ecfp_info(mols):
    #preallocate array and empty lists
    array = np.empty((len(mols), 1024), dtype=np.float32)
    info_box = []
    fps = [0]*len(mols)
    
    #iterate over all mols
    for i in range(len(mols)):
        #create empty dict, get ECFPs, store bitInfo in empty dict
        info = {}
        fps[i] = AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, 1024, bitInfo=info)
        
        #append i-th dict into general info list 
        info_box.append(info)
        
    #convert ECFPs to numpy array
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])    
    
    return array, info_box
    
#-----------------------------------------------------------------------------#    

def get_shap(model, ecfp):
    #create Explainer and calculate local shapley values for all ECFPs
    explainer = shap.Explainer(model)
    shaps = explainer(ecfp).values
    
    #depending on GBM,slice dataset
    if len(shaps.shape) == 3:
        shaps = shaps[:,:,0]
        
    #get global shaps and select top 20 vars
    shaps = np.abs(shaps)
    shaps = np.mean(shaps, axis=0)
    top_shaps = np.argsort(shaps)[-20:]
    
    return np.array(top_shaps, dtype = np.int16)
    
#-----------------------------------------------------------------------------#    

def process_shap(ecfp,
                 mols,
                 shap_idx,
                 info):
    
    #preallocate list of right size
    box = [0]*len(shap_idx)
    
    #iterate over all IDs of top 20 vars
    for i in range(len(shap_idx)):
        #get i-th bit
        target_idx = shap_idx[i]
        
        #find a mol with that bit set to 1
        mol_idx = np.where(ecfp[:, target_idx] == 1)[0][0]
        
        #store tuple for RDKIT fragment drawing function
        box[i] = (mols[mol_idx],
                  shap_idx[i],
                  info[mol_idx])
        
    return box
    
#-----------------------------------------------------------------------------#









