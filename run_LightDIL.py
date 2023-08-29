import os
from Dataset import *
import numpy as np
import torch
import torch.nn.functional as F
from models import LightDIL
import argparse
import random
from ray import tune
from ray.tune import CLIReporter
from load_models import LOAD_TEST
from configs import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def arg_para():
    parser = argparse.ArgumentParser(description='IRM-Feature-Interaction-Selection.')
    parser.add_argument('--k',type=int,default=48,help = 'dim of hidden layer of feature interaction')
    parser.add_argument('--patience',type=int,default=25,help = 'early stop epochs')
    parser.add_argument('--data_name',type= str,default='douban',help = 'name of the dataset, douban, ml-10m')
    parser.add_argument('--model_name',type= str,default='LightDIL',help = 'name of model(you cannot assign models through this arg)')
    parser.add_argument('--trial_name',type= str,default='0712',help = 'name of trial')
    parser.add_argument('--batch_size',type = int, default = 8192)
    parser.add_argument('--seed',type = int, default = 2000, help = 'random seed for each trial')
    return parser.parse_args()
args = arg_para()
print(args)

# search space for hyper-parameters
config = {
        'lr1':tune.grid_search([1e-3]),
        'update_lr':tune.grid_search([1]), 
        'tau':tune.grid_search([5]),  
        'lr2':tune.grid_search([1e-3]),
        'l2_w':tune.grid_search([1e-5]), 
        'inter_weight':tune.grid_search([1]), 
        'var':tune.grid_search([1]), 
        'init_unstable_weight':tune.grid_search([0.1,]),
        'compare_weight':tune.grid_search([1e-2])
    }


def trainable(config, checkpoint_dir=None):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    k = args.k
    data_name = args.data_name
    if args.data_name == 'douban':
        train_envs = [14,15,16,17,18]
        val_envs = [19,20,21,22,23]

    elif args.data_name == 'ml-10m':
        train_envs = [13,14,15,16,17]
        val_envs = [18,19,20,21]
    data = None
    Dataset = MyDataset(data_name, k,train_envs,val_envs, data)
    train_data = Dataset.train_data
    val_data = Dataset.val_data
    model_input = Dataset.model_input
    fixlen_feature_columns = Dataset.fixlen_feature_columns
    varlen_feature_columns = Dataset.varlen_feature_columns 
    val_model_input = Dataset.val_model_input
    target = Dataset.target
    linear_feature_columns =  varlen_feature_columns + fixlen_feature_columns
    dnn_feature_columns =  varlen_feature_columns + fixlen_feature_columns
    use_linear = True
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    loss_fn = F.binary_cross_entropy_with_logits
    model = LightDIL(linear_feature_columns,dnn_feature_columns,device = device
    ,interweight = config['inter_weight'],l2_reg_embedding=config['l2_w'],use_linear = use_linear
    , train_env_lst = train_envs,init_unstable_weight = config['init_unstable_weight'])
    var = [] # get embedding params and out bias in this list
    for name,param in model.named_parameters():
        if 'inter_weight' not in name and 'unstable_weight' not in name:
            var.append(param)
    opt1 = torch.optim.Adam(var, lr = config['lr1'])
    opt2 = torch.optim.Adam([
        {'params': [model.fm.inter_weight], 'lr': config['lr1']},# config['lr1']},
        {'params': model.fm.unstable_weight.parameters(), 'lr': config['lr2']},
    ])
    update_lr = config['update_lr']
    model.compile(opt1,opt2,update_lr,config['var'],loss_fn, metrics=['auc','logloss'] )
    model_config = {}
    for key,value in config.items():
        model_config[key] = config[key]
    model.get_model_info(args.data_name, args.model_name, model_config, args.trial_name)
    _ = model.fit(model_input,train_data[target].values,batch_size=args.batch_size,epochs=200,verbose=2,
    validation_data=(val_model_input,val_data[target]),train_env=True,train_env_lst=train_envs,
    val_env=True,val_env_lst=val_envs,tau=config['tau'],patience=args.patience, compare_weight = config['compare_weight'])
        

reporter = CLIReporter(
        metric_columns=["train_auc","val_auc_top","training_iteration",'val_auc'])
result = tune.run(
    trainable,
    resources_per_trial={"cpu": 1, "gpu": 0.5},
    local_dir = '/data/shith/ray_results',
    name = args.data_name + args.model_name + args.trial_name,
    config=config,
    progress_reporter=reporter,
    )

if args.data_name == 'douban':
    train_envs = [14,15,16,17,18]
    val_envs = [19,20,21,22,23]
    test_envs = [24,25,26,27,28]
elif args.data_name == 'ml-10m':
    train_envs = [13,14,15,16,17]
    val_envs = [18,19,20,21]
    test_envs = [22,23,24,25]

metrics = ['auc','logloss']
lt = LOAD_TEST(data_name = args.data_name, model_name = args.model_name, trial_name = args.trial_name)
lt.load_test(args.model_name + args.trial_name, test_envs, metrics = ['auc', 'logloss'])
lt.load_test(args.model_name + args.trial_name, val_envs, metrics = ['auc', 'logloss'], test_mode = 'val')
print("\n*************the best model found by ray[tune]:*************")
print("best result is:",result.get_best_trial('val_auc_top','max','last').last_result["val_auc_top"])

