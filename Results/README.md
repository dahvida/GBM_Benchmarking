# Results folder  
Contains the results presented in the original publication. Keep in mind that if you run any pipeline from the `Scripts` folder, you may overwrite these files.  

## Files
- `x_performance.csv`: contains the results of all GBM implementations for a given dataset "x". Columns are metrics, rows are replicates.   
- `fps.csv`: contains the bit density for ECFP fingerprints for each dataset using 1024, 2048 and 4096 bits.  
- `fungal_grid.csv`: contains the results for the hyperparameter grid analysis when varying the optimization grid. Rows are replicates, columns are different metrics. "d" denotes the complete grid, while "o" indicates the use of the grid from fANOVA. Numbers indicate the number of optimization iterations, while ECFP/MACCS/RDKIT indicate which representation was used. For example then, a column named "ECFP_ROC_100_d" indicates the ROC-AUC of a LightGBM model tuned with the default grid for 100 iterations using ECFP on the fungal dataset.  
- `sider_grid.csv`: same as `fungal_grid.csv`, but for the SIDER dataset.  
- `moldata_grid.csv`: same as `fungal_grid.csv`, but contains the results using ECFPs for all datasets from the Moldata benchmark.  
- `molnet_grid.csv`: same as `moldata_grid.csv`, but for the MoleculeNet benchmark.  
- `shap_comp.csv`: contains the top-20 overlap % for each dataset and each GBM pair.  
- `single_hyperparameters.csv`: contains the fANOVA importance scores for each hyperparameter across each endpoint modelled in Tox21, MUV, HIV, ClinTox, BACE, BBBP, NTP, Oxi and Phos datasets (55 endpoints in total).  
- `top10_interactions.csv`: contains the fANOVA importance scores for the top-10 first order interactions for the Tox21, MUV, HIV, ClinTox, BACE, BBBP, NTP, Oxi and Phos datasets (55 endpoints in total).  


