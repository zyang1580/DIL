import os
from Dataset import *
import numpy as np
import torch
import torch.nn.functional as F
from models import DIL
import argparse
import random
from ray import tune
from ray.tune import CLIReporter
from load_models import LOAD_TEST
from configs import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'# "0,3,5,6"

def arg_para():
    parser = argparse.ArgumentParser(description='IRM-Feature-Interaction-Selection.')
    parser.add_argument('--k',type=int,default=48,help = 'dim of hidden layer of feature interaction')
    parser.add_argument('--data_name',type= str,default='douban',help = 'name of the dataset, choose from: douban, ml-10m')
    parser.add_argument('--model_name',type= str,default='DIL',help = 'name of model(you cannot assign models through this arg)')
    parser.add_argument('--trial_name',type= str,default='0712',help = 'name of trial')
    parser.add_argument('--batch_size',type = int, default = 8192)
    parser.add_argument('--seed',type = int, default = 2000, help = 'random seed for each trial')
    parser.add_argument('--patience',type=int,default=25,help = 'early stop epochs')
    return parser.parse_args()
args = arg_para()
print(args)

# use to init model
config = {
        'lr1':1,
        'lr2':1,
        'update_lr':1,
        'l2_w':1,
        'l2_w_unstable':1,
        'var':1,
        'compare_weight':1,
        'tau':1
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
Dataset = MyDataset(data_name, k,train_envs,val_envs)
fixlen_feature_columns = Dataset.fixlen_feature_columns
varlen_feature_columns = Dataset.varlen_feature_columns 
linear_feature_columns =  varlen_feature_columns + fixlen_feature_columns
dnn_feature_columns =  varlen_feature_columns + fixlen_feature_columns
use_linear = True
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
loss_fn = F.binary_cross_entropy_with_logits
model = DIL(linear_feature_columns,dnn_feature_columns,device = device
,l2_reg_embedding=config['l2_w'],use_linear = use_linear
, train_env_lst = train_envs,l2_w_unstable = config['l2_w_unstable'])
embedding_params_id = list()
embedding_params_id.extend(list(map(id, model.embedding_dict_stable.parameters())))
embedding_params_id.extend(list(map(id,model.embedding_dict_unstable.parameters())))
other_params = list(filter(lambda p: id(p) not in embedding_params_id, model.parameters()))
opt2 = torch.optim.Adam([
    {'params': model.embedding_dict_stable.parameters(), 'lr': config['lr1']},# config['lr1']},
])
opt1 = torch.optim.Adam([
    {'params': model.embedding_dict_stable.parameters(), 'lr': config['lr1']},# config['lr1']},
    {'params': other_params, 'lr':config['lr1']}
])
opt3 = torch.optim.Adam([
    {'params': model.embedding_dict_unstable.parameters(), 'lr': config['lr2']},
])
update_lr = config['update_lr']# config['lr1']
model.compile(opt1,opt2,opt3,update_lr,config['var'],loss_fn, metrics=['auc','logloss'] )
model_config = {}
log_dir = os.path.join(workspace,'model_scale',args.data_name,args.model_name + args.trial_name)
if not os.path.exists(log_dir):
        os.makedirs(log_dir)
path_str = ''
for key,value in config.items():
    model_config[key] = config[key]
    path_str += (key + '=' + str(value) + '_')
path_str += '.json'
log_dir = os.path.join(log_dir,path_str)
model.get_model_info(args.data_name, args.model_name, model_config, args.trial_name)
if args.data_name == 'douban':
    file_path = model_param_path_DIL_douban
elif args.data_name == 'ml-10m':
    file_path = model_param_path_DIL_ml
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
lt.load_test(args.model_name + args.trial_name, test_envs, metrics = ['auc', 'logloss'])
lt.load_test(args.model_name + args.trial_name, val_envs, metrics = ['auc', 'logloss'], test_mode = 'val')




