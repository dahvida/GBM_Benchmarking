import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
from rdkit.Chem import MACCSkeys

###############################################################################

"""
PREPROCESSING FUNCTIONS

- get_ECFP:		computes ECFPs for molecule list
- get_MACCS:		computes MACCS for molecule list
- get_RDKIT:		computes 2D molecular descriptors from RDKIT for molecule list
- molnet_loader:	featurizes given task in given dataset from moleculenet
- moldata_loader:	splits / featurizes given task in given dataset from moldata
"""

#-----------------------------------------------------------------------------#

def get_ECFP(mols, radius = 2, nbits = 1024):
    #initialize empty array
    array = np.empty((len(mols), nbits), dtype=np.float32)
    
    #get FPs as list
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius, nbits) for x in mols]
    
    #convert to numpy array
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])
        
    return array

#-----------------------------------------------------------------------------#

def get_MACCS(mols):
    #initialize empty array
    array = np.empty((len(mols), 167), dtype=np.float32)
    
    #get FPs as list
    fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
    
    #convert to numpy array
    for i in range(len(array)):
        DataStructs.ConvertToNumpyArray(fps[i], array[i])
        
    return array
    
#-----------------------------------------------------------------------------#    

def get_RDKIT(mols):
    #load list of 2D descriptors and create Calculator
    names = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    
    #initialize empty array
    descs = np.empty((len(mols), 208))
    
    #compute descriptors for each molecule
    for i in range(len(descs)):
        descs[i] = calc.CalcDescriptors(mols[i])
        
    #deal with numerical issues
    descs = np.nan_to_num(descs, posinf=10e10, neginf=-10e10)
    
    return descs

#-----------------------------------------------------------------------------#

def molnet_loader(dataset_name, task_id, fp_type = "ECFP"):
    #load dataset from .csv file in ../Datasets
    dataset_name = dataset_name.lower()
    db = pd.read_csv("../Datasets/" + dataset_name + ".csv")
    
    #get relevant column and drop compounds that don't have a label
    task_name = list(db.columns)[task_id]
    db = db.loc[:, ["smiles", task_name]]    
    db.dropna(inplace=True)
    
    #get mols for dataset
    smiles = list(db["smiles"])
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    
    #depending on fp_type, compute representation
    if fp_type == "ECFP":
        x = get_ECFP(mols)
    elif fp_type == "MACCS":
        x = get_MACCS(mols)
    elif fp_type == "RDKIT":
        x = get_RDKIT(mols)
    else:
        x = mols
        
    #get labels as numpy array
    y = np.array(db[task_name])
    
    return x, y

#-----------------------------------------------------------------------------#

def moldata_loader(dataset_name, task_id, fp_type = "MACCS"):
    #load dataset from .csv file in ../Datasets
    dataset_name = dataset_name.lower()
    db = pd.read_csv("../Datasets/" + dataset_name + ".csv")
    
    #get relevant column and drop compounds that don't have a label
    task_name = list(db.columns[2:-1])[task_id]
    db = db.loc[:, ["smiles", task_name, "split"]]
    db.dropna(inplace=True)
    
    #find train/test/val splits from Arshadi et al
    train = db.loc[db["split"]=="train"]
    test = db.loc[db["split"]=="test"]
    val = db.loc[db["split"]=="validation"]
    
    #get mols and ys for train, test and val sets
    X_tr = list(train["smiles"])
    X_tr = [Chem.MolFromSmiles(x) for x in X_tr]
    y_tr = np.array(train[task_name])
        
    X_te = list(test["smiles"])
    X_te = [Chem.MolFromSmiles(x) for x in X_te]
    y_te = np.array(test[task_name])
        
    X_v = list(val["smiles"])
    X_v = [Chem.MolFromSmiles(x) for x in X_v]
    y_v = np.array(val[task_name])
    
    #depending on fp_type, compute representation for all sets
    if fp_type == "ECFP":
        X_tr = get_ECFP(X_tr)
        X_te = get_ECFP(X_te)
        X_v = get_ECFP(X_v)
    elif fp_type == "MACCS":
        X_tr = get_MACCS(X_tr)
        X_te = get_MACCS(X_te)
        X_v = get_MACCS(X_v)
    elif fp_type == "RDKIT":
        X_tr = get_RDKIT(X_tr)
        X_te = get_RDKIT(X_te)
        X_v = get_RDKIT(X_v)
    else:
        pass
    
    return X_tr, y_tr, X_te, y_te, X_v, y_v       
    
#-----------------------------------------------------------------------------#
    
    


