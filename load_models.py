import os
from configs import *
from Dataset import MyDataset
import torch
from models import *
import argparse
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def arg_para():
    parser = argparse.ArgumentParser(description='IRM-Feature-Interaction-Selection.')
    parser.add_argument('--batch_size', type=float, default=8192, help='batch size')
    parser.add_argument('--data_name',type= str,default='douban',help = 'name of the dataset')
    parser.add_argument('--model_name',type= str,default = None,help = 'name of the dataset')
    parser.add_argument('--trial_name',type= str,default = None,help = 'name of the dataset')
    parser.add_argument('--seed',type= int,default=2000,help = 'random seed')
    parser.add_argument('--k',type=int,default=48,help = 'dim of hidden layer of feature interaction')
    return parser.parse_args()


class LOAD_TEST():
    def __init__(self, data_name, model_name = None, trial_name = None, dir = None, model = None):
        """
        data_name is necessary
        you can assign data_name, model_name, trial_name, or only assign the dir to get models
        the dir will be like workspace/data_name/model_name/trial_name 
        """
        # self.Dataset = MyDataset(data_name, k, test_envs)
        self.data_name = data_name
        checkpoint_dir = os.path.join(workspace, data_name)
        if model_name is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if trial_name is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, trial_name)
        if dir is not None:
            checkpoint_dir = dir
        self.checkpoint_dir = checkpoint_dir
        
        # load saved model
        if model is None:
            file_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            model = torch.load(file_path)['model']
            self.model = model
        # direct load model
        else:
            self.model = model
            
    def load_test(self, log_name,  test_envs, k = 48, metrics = ['auc'], batch_size = 8192, test_mode = 'test',data = None):
        Dataset = MyDataset(self.data_name, k, val_envs = test_envs, data = data) 
        print('loading best models in validation sets and testing...')
        # file_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        model = self.model
        model.metrics = model._get_metrics(metrics)
        # Dataset.update_val_env(test_envs)
        val_data = Dataset.val_data
        val_model_input = Dataset.val_model_input
        user_id = Dataset.user_id
        target = Dataset.target
        user_column_name  = Dataset.user_column_name
        # actually they are test_data,test_model_input...
        try:
            eval_result = model.evaluate2(x = val_model_input,y = val_data[target],batch_size=batch_size,val_env = True,
            val_env_lst = test_envs, user_id = user_id, u = val_data[user_column_name])
        except:
            eval_result = model.evaluate2(x = val_model_input,y = val_data[target],batch_size=batch_size,val_env = True,
            val_env_lst = test_envs, user_id = user_id, u = None)
        if 'test' in test_mode:
            log_dir = os.path.join(workspace, 'test_results', self.data_name)
            log_path = os.path.join(log_dir, log_name + '.json')
        elif 'val' in test_mode:
            log_dir = os.path.join(workspace, 'val_results', self.data_name)
            log_path = os.path.join(log_dir, log_name + '.json')
        elif 'train' in test_mode:
            log_dir = os.path.join(workspace, 'train_results', self.data_name)
            log_path = os.path.join(log_dir, log_name + '.json')
        file_path = os.path.join(self.checkpoint_dir, 'best_result.pt')
        configs = torch.load(file_path)['model_config'] 
        print(eval_result)
        print(configs)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file = open(log_path, "w")
        json.dump(eval_result, file)
        file.write('\r\n')
        light_result = {}
        for key,value in eval_result.items():
            if ('aver' in key): # or ('var' in key):
                light_result[key] = value
        json.dump(light_result, file)
        file.write('\r\n')
        light_result = {}
        for key,value in eval_result.items():
            if ('var' in key): # or ('var' in key):
                light_result[key] = value
        json.dump(light_result, file)
        file.write('\r\n')
        model_config = model.model_config_dict
        json.dump(model_config, file)
        file.close()     
        return 


