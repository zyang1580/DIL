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


# use to init model
config = {
        'lr1':1,
        'update_lr':1,
        'tau':1,
        'lr2':1,
        'l2_w':1 ,
        'inter_weight':1, 
        'var':1, 
        'init_unstable_weight':1,
        'compare_weight':1
    }


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


if args.data_name == 'douban':
    file_path = model_param_path_LightDIL_douban
elif args.data_name == 'ml-10m':
    file_path = model_param_path_LightDIL_ml
model_state_dict = torch.load(file_path)
print(list(model_state_dict.keys()))
model.load_state_dict(model_state_dict)
        


if args.data_name == 'douban':
    train_envs = [14,15,16,17,18]
    val_envs = [19,20,21,22,23]
    test_envs = [24,25,26,27,28]
 
elif args.data_name == 'ml-10m':
    train_envs = [13,14,15,16,17]
    val_envs = [18,19,20,21]
    test_envs = [22,23,24,25]

metrics = ['auc','logloss']
lt = LOAD_TEST(data_name = args.data_name, model_name = args.model_name, trial_name = args.trial_name, model = model)
lt.model = model
lt.load_test(args.model_name + args.trial_name, test_envs, metrics = ['auc', 'logloss'])
lt.load_test(args.model_name + args.trial_name, val_envs, metrics = ['auc', 'logloss'], test_mode = 'val')

