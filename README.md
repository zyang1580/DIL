# DIL

This is an implementation for our SIGIR 2023 paper "Reformulating CTR Prediction: Learning Invariant Feature Interactions for Recommendation" based on PyTorch.
This Repo contains the required code to run DIL and LighDIL on Douban, ML-10M datasets.

## Requirement
To use this code, ensure you have the following prerequisites:
- Python 3.x
- PyTorch
- numpy
- pandas
- sklearn
- ray
- deepctr_torch
- etc..

## Key parameters:
- --lr1: learning rate for most parameters
- --lr2: learning rate for \phi_t
- --update_lr: learning rate for meta training
- --l2_w: L2 regularization cofficient for model parameters
- --l2_w_unstable: L2 regularization cofficient for phi_t in DIL
- --var: coefficient controlling the variance auxiliary loss
- --tau: temperature coefficient controlling the softmax weighting of the environment loss
- --compare_weight: coefficient controlling the L_int."


## Project description
This repository contains the following files:
- configs.py: Define file paths for models. Please update these paths accordingly.
- run_DIL.py, run_LightDIL.py: Main scripts for running DIL and LightDIL, respectively, with hyperparameter search capabilities.
- run_quick_DIL.py, run_quick_LightDIL.py: Quick versions of the scripts for fast model reproduction by loading pre-trained parameters.

To execute the code, you can download the required datasets and model parameters from https://drive.google.com/drive/folders/1_67rx6vLShp7mFvP7Ku2SLk61ie9Ru-6?usp=drive_link


## Running example
python run_quick_DIL.py --data_name 'ml-10m' --trial_name '0719' --seed 2000
python run_quick_LightDIL.py --data_name 'douban' --trial_name '0719' --seed 2000 

python run_LightDIL.py --data_name 'douban' --trial_name '0719' --seed 2000 
python run_DIL.py --data_name 'ml-10m' --trial_name '0719' --seed 2000
- To search hyper-parameters, execute the code "run_DIL.py" or "run_LightDIL.py". The search space is provided by the following code.
``` 
# config for DIL
config = {
        'lr1':tune.grid_search([1e-3]),
        'lr2':tune.grid_search([1e-3]),
        'update_lr':tune.grid_search([1,1e-1,1e-2,1e-3]),
        'l2_w':tune.grid_search([1e-3,1e-4,1e-5,1e-6,1e-7]),
        'var':tune.grid_search([1,1e-1,1e-2,1e-3]),
        'compare_weight':tune.grid_search([1e-2,1e-3]),
        'tau':tune.grid_search([0,5,20,40,]),
        'l2_w_unstable':tune.grid_search([1e-3,5e-3,1e-4]),
    }

# config for LightDIL
config = {
        'lr1':tune.grid_search([1e-3]),
        'lr2':tune.grid_search([1e-3]),
        'update_lr':tune.grid_search([1,1e-1,1e-2,1e-3]), 
        'l2_w':tune.grid_search([1e-3,1e-4,1e-5,1e-6,1e-7]), 
        'var':tune.grid_search([1,1e-1,1e-2,1e-3]), 
        'compare_weight':tune.grid_search([1e-2,1e-3])，
        'tau':tune.grid_search([0,5,20,40,]),  
        'inter_weight':tune.grid_search([1,0.1,0.01]), 
        'init_unstable_weight':tune.grid_search([1,0.1,0.01]),
    }
```
python run_quick_LightDIL.py --data_name 'douban' --trial_name '0719' --seed 2000 


## Citation

ACM ref:

>Yang Zhang, Tianhao Shi, Fuli Feng, Wenjie Wang, Dingxian Wang, Xiangnan He, and Yongdong Zhang. 2023. Reformulating CTR Prediction: Learning Invariant Feature Interactions for Recommendation. ACM SIGIR'23. https://doi.org/10.1145/3539618.3591755

