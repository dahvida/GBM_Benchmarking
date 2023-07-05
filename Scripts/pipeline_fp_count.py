from loaders import *
import numpy as np
import pandas as pd
from rdkit import Chem
import os

def load_data(path):
    db = pd.read_csv(path, index_col=0)
    mols = [Chem.MolFromSmiles(x) for x in list(db["smiles"])]
    
    return mols

def process_mols(mols):
    first = get_ECFP(mols, nbits=1024)
    first = np.mean(np.sum(first, axis=1))
    second = get_ECFP(mols, nbits=2048)
    second = np.mean(np.sum(second, axis=1))
    third = get_ECFP(mols, nbits=4096)
    third = np.mean(np.sum(third, axis=1))
    
    return first, second, third

def main():
    paths = os.listdir("../Datasets")
    names = [x[:-4] for x in paths]
    results = np.zeros((17,3))
    for i in range(len(paths)):
        print("Processing:", names[i])
        mols = load_data(paths[i])
        results[i,0], results[i,1], results[i,2] = process_mols(mols)
    
    db = pd.DataFrame({
        "Dataset": names,
        "1024": results[:,0],
        "2048": results[:,1],
        "4096": results[:,2]
        })
    
    db.to_csv("fps.csv")

if __name__ == "__main__":
    main()
