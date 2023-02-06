# GBM Benchmarking  
![Alt text](/Pictures/graphical_abstract.png)
![Python 3.6](https://img.shields.io/badge/python-3.7%20%7C%203.8-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
Repository containing the code and the datasets to reproduce the results from the paper "Practical guidelines for the use of Gradient Boosting for molecular property prediction".  

## Installation  
All necessary packages can be installed via conda from the environment.yml file.  
```
git clone https://github.com/dahvida/GBM_Benchmarking
conda env create --name GBM --file=environments.yml
conda activate GBM
```

## Usage
All results can be reproduced by executing the respective *_pipeline.py files in the Scripts folder. The outputs from each script can be found in the Results folder, either as .csv, .pkl or .txt files.  
- **pipeline_script.py**: &ensp;returns ROC-AUC, PR-AUC, training time and Shapley overlap for all GBM implementations on all datasets  
- **pipeline_hyperparam.py**: &ensp;evaluates the importance of each hyperparameter using fANOVA  
- **pipeline_grid.py**: &ensp;evaluates the performance of the grid with the most important hyperparameters versus optimizing all possible hyperparameters  
- **pipeline_fragments.py**: &ensp;draws the top 20 most important ECFP bits for the BACE dataset for all GBM implementations  
- **pipeline_reproducibility.py**: &ensp;evaluates the Shapley overlap across all datasets for two independent optimization and training runs with LightGBM  

For example, here is the code to execute the script for obtaining the classification performance results using the hyperparameters used in the paper:  
```
cd ./GBM_Benchmarking/Scripts
python3 pipeline_benchmarking.py
```


