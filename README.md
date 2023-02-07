# GBM Benchmarking  
![Alt text](/Pictures/graphical_abstract.png)
![Python 3.6](https://img.shields.io/badge/python-3.7%20%7C%203.8-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
Repository containing the code and the datasets to reproduce the results from the paper "*Practical guidelines for the use of Gradient Boosting for molecular property prediction*".  

## Repository structure
- **Datasets:** Contains all datasets used in this study as .csv files  
- **Scripts:** Contains all scripts and utility functions used to reproduce the results  
- **Results:** Contains the outputs from the pipeline functons  

## Installation  
All necessary packages can be installed via conda from the environment.yml file.  
```
git clone https://github.com/dahvida/GBM_Benchmarking
conda env create --name GBM --file=environments.yml
conda activate GBM
```

## Usage
All results can be reproduced by executing the respective *pipeline_x.py* files in the *Scripts* folder. The outputs from each script will be saved in the *Results* folder, either as .csv, .pkl or .txt files.  
- **pipeline_benchmark.py:** Returns ROC-AUC, PR-AUC, training time and Shapley overlap for all GBM implementations on the selected datasets  
- **pipeline_hyperparam.py:** Evaluates the importance of each hyperparameter using fANOVA on the selected datasets  
- **pipeline_grid.py:** Evaluates the performance of the grid with the most important hyperparameters versus optimizing all possible hyperparameters  
- **pipeline_fragments.py:** Draws the top 20 most important ECFP bits for a given dataset for all GBM implementations. Currently supports only the BACE, BBBP and HIV datasets  
- **pipeline_reproducibility.py:** Evaluates the Shapley overlap on the selected datasets for two independent optimization and training runs with LightGBM  

## Tutorial
Each script uses as default arguments the same parameters used in the paper. For example,here is the code to execute the script for obtaining the classification performance results:  
```
cd /GBM_Benchmarking/Scripts
python3 pipeline_benchmarking.py
```
Here is the code to execute the same script, changing the number of iterations for optimization and evaluation:  
```
cd /GBM_Benchmarking/Scripts
python3 pipeline_benchmarking.py --opt_iters 30 --iters_moleculenet 10 --iters_moldata 3
```
Check the *pipeline_x.py* files in the *Scripts* folder for a description of the available arguments for each script.  

## How to cite
Link to publication  


