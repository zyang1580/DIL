import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from configs import *
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models.deepfm import *
import time
from deepctr_torch.layers.utils import slice_arrays
from layers import *
from ray import tune
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
import time
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from deepctr_torch.layers import PredictionLayer
from deepctr_torch.callbacks import History
import json
import pandas as pd
def varlen_embedding_lookup_fast(X, embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
        varlen_embedding_vec_dict = {}
        for fc in varlen_sparse_feature_columns:
            feature_name = fc.name
            embedding_name = fc.embedding_name
            if fc.use_hash:
                # lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
                # TODO: add hash function
                lookup_idx = sequence_input_dict[feature_name]
            else:
                lookup_idx = sequence_input_dict[feature_name]
            varlen_embedding_vec_dict[feature_name] = F.embedding(input = X[:, lookup_idx[0]:lookup_idx[1]].long(),
                weight = embedding_dict[embedding_name])
                 # (lookup_idx)
        return varlen_embedding_vec_dict

class my_early_stoper(object):
    def __init__(self,refer_metric='val_aver_auc',stop_condition = 5):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric
        print("*****stoper setting:",stop_condition)

    def update_and_isbest(self,eval_metric,epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        else:
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]: # update the best results
                self.best_eval_result = eval_metric
                self.not_change = 0
                self.best_epoch = epoch
                return True              # best
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            return False

class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super(BaseModel, self).__init__()
        #torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

    def get_model_info(self, data_name, model_name, configs, trial_name = None):
        model_config_name = model_name
        for key,value in configs.items():
            model_config_name += (',' + key + ':' + str(value))
        self.data_name = data_name
        self.model_name = model_name
        self.trial_name = trial_name
        self.model_config_name = model_config_name
        self.model_config_dict = configs

    def evaluate(self, x, y, batch_size=256,u=None):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result


    def predict(self, x, batch_size=256, ):
        """
        
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
            x_temp.append(x['env'])
            x = x_temp
            for i in range(len(x)):
                if len(x[i].shape) == 1:
                    x[i] = np.expand_dims(x[i], axis=1)
            x = np.concatenate(x, axis=-1)
        tensor_data = Data.TensorDataset(
            torch.from_numpy(x))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)
        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x_ = x_test[0].to(self.device).float()
                y_pred = model(x_)
                y_pred = y_pred.cpu().data.numpy() 
                pred_ans.append(y_pred)
        return np.concatenate(pred_ans).astype("float64") 


    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns] #error

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)
        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)
    
    def _log_loss_with_logits(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        loss_func = F.binary_cross_entropy_with_logits
        with torch.no_grad():
            y_true = torch.from_numpy(y_true).float()# .reshape(-1,1)
            y_pred = torch.from_numpy(y_pred).float()# .reshape(-1,1)
            loss = loss_func(y_pred.reshape(-1,1),y_true.reshape(-1,1),reduction='mean')
        return float(loss.clone().detach().numpy().tolist())

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    metrics_[metric] = self._log_loss_with_logits
                # if metric == "binary_crossentropy" or metric == "logloss":
                    # if set_eps:
                    #     metrics_[metric] = self._log_loss
                    # else:
                    #     metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                if metric =='gauc' or metric == 'uauc':
                    metrics_[metric] = self.cal_group_auc
                self.metrics_names.append(metric)
        return metrics_

    def cal_group_auc(self,labels, preds, user_id_list):
        """Calculate group auc"""
        
        print('*' * 50)
        if len(user_id_list) != len(labels):
            raise ValueError(
                "impression id num should equal to the sample num," \
                "impression id num is {0}".format(len(user_id_list)))
        group_score = defaultdict(lambda: [])
        group_truth = defaultdict(lambda: [])
        for idx, truth in enumerate(labels):
            user_id = user_id_list[idx]
            score = preds[idx]
            truth = labels[idx]
            group_score[user_id].append(score)
            group_truth[user_id].append(truth)

        group_flag = defaultdict(lambda: False)
        for user_id in set(user_id_list):
            truths = group_truth[user_id]
            flag = False
            for i in range(len(truths) - 1):
                if truths[i] != truths[i + 1]:
                    flag = True
                    break
            group_flag[user_id] = flag

        impression_total = 0
        total_auc = 0
    #
        for user_id in group_flag:
            if group_flag[user_id]:
                auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
                total_auc += auc * len(group_truth[user_id])
                impression_total += len(group_truth[user_id])
        group_auc = float(total_auc) / impression_total
        group_auc = round(group_auc, 4)
        return group_auc
    def evaluate2(self, x, y, batch_size=256,val_env = True,val_env_lst = [],user_id = None,u = None):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        self.eval()
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
            x_temp.append(x['env'])
            x = x_temp
        if isinstance(y, pd.DataFrame):
            y = y.values
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = np.concatenate(x, axis=-1)
        if(val_env == True):
            batch_env = x[:,-1]
            validate_dict = {}
            eval_result = {}
            for name, metric_fun in self.metrics.items():
                validate_dict[name] = np.zeros(len(val_env_lst))
            pred_ans = self.predict(x, batch_size=batch_size)
            for i,env_num in enumerate(val_env_lst):
                y_env = y[batch_env == env_num]
                if user_id is not None:
                    user_id_env = user_id[batch_env == env_num]
                pred_ans_env = pred_ans[batch_env == env_num]
                for name, metric_fun in self.metrics.items():
                    if(name == 'gauc' or name == 'uauc'):
                        eval_result[name + '_env_' +str(env_num)] = metric_fun(y_env, pred_ans_env,user_id_env)
                    else:
                        eval_result[name + '_env_' +str(env_num)] = metric_fun(y_env, pred_ans_env)
                    validate_dict[name][i] = eval_result[name + '_env_' +str(env_num)]
            for name, metric_fun in self.metrics.items():
                eval_result['aver_' + name] = np.average(validate_dict[name])
                eval_result['var_' + name] = np.var(validate_dict[name],ddof=1)   
                if(name == 'auc' or name == 'logloss'):
                    eval_result[name] = metric_fun(y, pred_ans)
                if(name == 'gauc' or name == 'uauc'):
                    eval_result[name] = metric_fun(y, pred_ans,user_id)
        if(val_env == False):
            pred_ans = self.predict(x, batch_size,)
            eval_result = {}
            for name, metric_fun in self.metrics.items():
                if(name == 'gauc' or name == 'uauc'):
                    eval_result[name] = metric_fun(y, pred_ans,user_id)
                else:
                    eval_result[name] = metric_fun(y, pred_ans)
        return eval_result
    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]


class LightDIL(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                  task='binary', device='cpu', gpus=None,use_linear = True
                  ,use_interweight = True,interweight = 1,l2_reg_inter_weight=0, train_env_lst = [],init_unstable_weight = 1):

        super().__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.use_linear = use_linear

        if use_fm :
            self.fm = Disentangle_FM(device, use_interweight, len(dnn_feature_columns),  
            interweight, train_env_lst, init_unstable_weight = init_unstable_weight)
            if(use_interweight == True):
                self.add_regularization_weight([self.fm.inter_weight], l2=l2_reg_inter_weight)
        self.to(device)

    def compile(self, optimizer1,optimizer2 = None,update_lr = 1e-3,
        hyper_aux_var_loss = 1e-1,loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim1 = self._get_optim(optimizer1) #use to update embeddings
        if optimizer2 is not None:
            self.optim2 = self._get_optim(optimizer2) #use meta loss to update inter_weight
        self.update_lr = update_lr 
        self.hyper_aux_var_loss = hyper_aux_var_loss
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)
    
    def forward(self, X, fast_weight = None,):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        
        if(self.use_linear): 
            logit = self.linear_model(X)
        else:
            logit = 0
        
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            
            logit += self.fm(fm_input, fast_weight)


        y_pred = logit + self.bias
        
        return y_pred

    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, train_env = False, train_env_lst = [],
            val_env = True, val_env_lst = [],user_id = None, save_models = True, tau = 20,patience=5, compare_weight = 1e-2):
        """
        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`
        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        print("runing stable FM with field level weight -- Version 2")
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
            x_temp.append(x['env']) # adding the env to the input list
            x = x_temp
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x_temp = [val_x[feature] for feature in self.feature_index]
                val_x_temp.append(val_x['env'])
                val_x = val_x_temp
        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        model = self.train()
        loss_func = self.loss_func
        optim1 = self.optim1  #use to update embeddings
        optim2 = self.optim2  #use to update inter_weight
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size,num_workers=4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        var = [] # get embedding params and out bias in this list
        for name,param in self.named_parameters():
            if 'inter_weight' not in name and 'unstable_weight' not in name:
                var.append(param)
        optim2.zero_grad()
        val_auc_top = 0
        val_logloss_min = 100
        stopers = my_early_stoper(refer_metric = 'val_aver_auc', stop_condition = patience)
        train_env_num = len(train_env_lst)
        
        """train models"""
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for batch_num, (x_train, y_train) in t:
                        x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                        y = y_train.to(self.device).float()
                        """update parameters except \phi"""
                        
                        y_pred = model(x).squeeze()
                        batch_env = x[:,-1] # envs
                        
                        env_idx_dict = {}
                        for env_x in train_env_lst:
                            env_idx_dict[env_x] = (batch_env == env_x).nonzero().squeeze()
                        
                        meta_loss = 0
                        env_i = train_env_lst[batch_num%train_env_num]
                        y_pred = model(x).reshape(-1,1)
                        y_true = y.reshape(-1,1)# y_true_i = y[batch_env == env_i].reshape(-1,1)
                        loss_env_emb = loss_func(y_pred, y_true,reduction = 'mean')#  reduction = 'sum') / batch_size
                        reg_loss = self.get_regularization_loss()
                        total_loss = reg_loss + loss_env_emb + self.aux_loss
                        optim1.zero_grad()
                        total_loss.backward(inputs = var) # var consists of params expect \phi
                        optim1.step()

                        """meta-learning, update stable \phi_s"""
                        x_env_i = x[env_idx_dict[env_i]]
                        sum_weight = self.fm.inter_weight + self.fm.unstable_weight[str(env_i)] 
                        y_pred_i = model(x_env_i, sum_weight).reshape(-1,1)
                        y_true_i = y[env_idx_dict[env_i]].reshape(-1,1) # empty # 
                        loss_env_i = loss_func(y_pred_i, y_true_i, reduction = 'mean')
                        grad_var = [sum_weight, self.fm.unstable_weight[str(env_i)]]
                        grad = torch.autograd.grad(loss_env_i, grad_var, create_graph = True)
                        sum_grad = grad[0]
                        fast_stable_weight = self.fm.inter_weight - self.update_lr * sum_grad
                        meta_loss_env_js_list = []
                        for env_j in train_env_lst:
                            if env_i == env_j:
                                continue
                            fast_envj_weight = fast_stable_weight + self.fm.unstable_weight[str(env_j)]
                            y_pred_j = model(x[env_idx_dict[env_j]], fast_envj_weight)
                            y_true_j = y[env_idx_dict[env_j]].reshape(-1,1)
                            single_meta_loss = loss_func(y_pred_j,y_true_j,reduction = 'mean') 
                            meta_loss_env_js_list.append(single_meta_loss.unsqueeze(-1))
                        meta_loss_env_js_list = torch.cat(meta_loss_env_js_list,dim=0)
                        meta_loss += self.hyper_aux_var_loss * torch.std(meta_loss_env_js_list,unbiased=False,)
                        with torch.no_grad():
                            meta_weight = meta_loss_env_js_list.detach()
                            meta_weight = F.softmax(meta_weight * tau, dim=0)
                        meta_loss += torch.mul(meta_weight, meta_loss_env_js_list).sum()                         
                        optim2.zero_grad()
                        meta_loss.backward(inputs = [self.fm.inter_weight])
                        optim2.step()


                        """update environment-specific part \phi_t"""
                        meta_loss = 0
                        env_self_loss = 0
                        for env_k in train_env_lst:
                            x_env_k = x[env_idx_dict[env_k]]
                            sum_weight = self.fm.inter_weight.detach() + self.fm.unstable_weight[str(env_k)] # no grad for stable-weight
                            y_pred_k = model(x_env_k, sum_weight).reshape(-1,1)
                            y_true_k = y[env_idx_dict[env_k]].reshape(-1,1) # empty # 
                            env_self_loss += loss_func(y_pred_k, y_true_k, reduction = 'mean')
                        y_pred_by_i = model(x, self.fm.inter_weight.detach() + self.fm.unstable_weight[str(env_i)]) # prediction by stable + env i
                        meta_loss_env_js_list = []
                        compare_loss = 0
                        for env_j in train_env_lst:
                            y_j_pred_by_i = y_pred_by_i[batch_env==env_j].squeeze()
                            x_env_j = x[batch_env==env_j]
                            y_true_j = y[batch_env == env_j].squeeze()
                            if env_j != env_i:
                                y_j_pred_by_j = model(x_env_j,self.fm.inter_weight.detach() + self.fm.unstable_weight[str(env_j)]).squeeze() # no grad
                                compare_loss += self.get_compare_loss(y_j_pred_by_j, y_j_pred_by_i, y_true_j)
                        optim2.zero_grad()
                        (env_self_loss + compare_loss * compare_weight).backward()
                        optim2.step()
                        
                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if(name == 'uauc' or name == 'gauc'):
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                print('train finished')
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            """test validation performance"""
            
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate2(val_x, val_y, batch_size,val_env,val_env_lst,user_id) # 
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                for name in self.metrics:
                    if(name == 'gauc' or name == 'uauc'):
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])
                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                print(eval_str)
            
            """save models and report valid performance"""

            need_saving = stopers.update_and_isbest(epoch_logs, epoch)
            need_stopping = stopers.is_stop()
            if self.trial_name is not None:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name, self.trial_name)
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt') 
            else:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name)
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                    'best_result.pt') 
            if not os.path.exists(mycheckpoint_dir):
                os.makedirs(mycheckpoint_dir)
            if not os.path.exists(mycheckpoint_path):
                best_result = 0
            else:
                best_result = torch.load(mycheckpoint_path)['result']
            if need_saving and save_models and best_result <= epoch_logs['val_aver_auc']:
                best_result = epoch_logs['val_aver_auc']
                if self.trial_name is not None:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt') 
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                    'best_result.pt') 
                mycheckpoint = {
                    'result':epoch_logs['val_aver_auc'],
                    'model_config':self.model_config_name
                }
                torch.save(mycheckpoint,mycheckpoint_path) 
                mycheckpoint = {
                    'epoch':epoch,
                    'model':self
                }
                if self.trial_name is not None:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_model.pt') # use folder name to distinguish val_env
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                     'best_model.pt') # use folder name to distinguish val_env
                        # self.model_config_name + 'epoch:' +str(epoch) + '.pt')
                torch.save(mycheckpoint,mycheckpoint_path)
            loss = epoch_logs['logloss']
            auc = epoch_logs['auc']
            val_auc = epoch_logs['val_aver_auc']
            val_logloss = epoch_logs['val_aver_logloss']
            if(val_auc>val_auc_top):
                val_auc_top = val_auc
            if(val_logloss_min > val_logloss):
                val_logloss_min = val_logloss
            if((train_env == True)):
                tune.report(val_auc_top = val_auc_top,val_logloss_min = val_logloss_min
            ,val_auc = val_auc,val_logloss = val_logloss,train_loss=loss,train_auc = auc,training_iteraction=epoch)
            else:
                tune.report(val_auc_top = val_auc_top,val_logloss_min = val_logloss_min
            ,val_auc = val_auc,val_logloss = val_logloss,train_loss=loss,train_auc = auc,training_iteraction=epoch)
            if need_stopping:
                print("early stop.......")
                break
            callbacks.on_epoch_end(epoch, epoch_logs)
        if self.stop_training:
            callbacks.on_train_end()
        return self.history


    def get_compare_loss(self, y_1, y_2, y_true):
        '''y_1 > y_2'''
        cmp_loss = y_true * F.relu(y_2 - y_1 ) + (1 - y_true) * F.relu(y_1 - y_2 )
        cmp_loss = torch.mean(cmp_loss)
        return cmp_loss





class DIL(BaseModel): # disentangle_embedding
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                  task='binary', device='cpu', gpus=None,use_linear = True
                  ,use_interweight = False,interweight = 1,l2_reg_inter_weight=0, train_env_lst = [], init_unstable_weight = 1,
                  l2_w_unstable = 1e-5,init_unstable = 0.0001):
        nn.Module.__init__(self)
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus

        self.use_fm = use_fm
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.use_linear = use_linear
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict_stable = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )
        self.embedding_dict_unstable = nn.ModuleDict({})
        for env in train_env_lst:
            self.embedding_dict_unstable[str(env)] = create_embedding_matrix(dnn_feature_columns, init_unstable, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict_stable.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)
        self.add_regularization_weight(self.embedding_dict_unstable.parameters(),l2 = l2_w_unstable)
        # self.out = PredictionLayer(task, )
        self.to(device)
        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()
        if use_fm :
            self.fm = FM(device,use_interweight,len(dnn_feature_columns), )
            if(use_interweight == True):
                self.add_regularization_weight([self.fm.inter_weight], l2=l2_reg_inter_weight)
        self.to(device)

    def compile(self, optimizer1,optimizer2 = None,optimizer3 = None,update_lr = 1e-3
                , hyper_aux_var_loss = 1e-1,loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim1 = self._get_optim(optimizer1) #use to update embeddings
        if optimizer2 is not None:
            self.optim2 = self._get_optim(optimizer2) #use meta loss to update inter_weight
        if optimizer3 is not None:
            self.optim3 = self._get_optim(optimizer3) #use meta loss to update inter_weight
        self.update_lr = update_lr 
        self.hyper_aux_var_loss = hyper_aux_var_loss
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns] #error

        sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)

        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list
        
    def input_from_feature_columns_fast(self, X, feature_columns, embedding_dict, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
        sparse_embedding_list = [F.embedding(input = X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long(),
            weight = embedding_dict[feat.embedding_name]) for feat in sparse_feature_columns] #error
        sequence_embed_dict = varlen_embedding_lookup_fast(X, embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]
        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def forward(self, X, fast_embedding_dict = None, fast_embedding_dict_unstable = None, env_num = None,):
        if(self.use_linear): 
            logit = self.linear_model(X)
        else:
            logit = 0
        if fast_embedding_dict is None:
            if env_num is not None:
                sparse_embedding_list1, dense_value_list1 = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict_stable)
                sparse_embedding_list2, dense_value_list2 = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict_unstable[str(env_num)])
                if self.use_fm and len(sparse_embedding_list1) > 0:
                    fm_input1 = torch.cat(sparse_embedding_list1, dim=1)
                    fm_input2 = torch.cat(sparse_embedding_list2, dim=1)
                    logit += self.fm(fm_input1+fm_input2) 
            else:
                sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict_stable)
                if self.use_fm and len(sparse_embedding_list) > 0:
                    fm_input = torch.cat(sparse_embedding_list, dim=1)
                    logit += self.fm(fm_input)
        else:
            if fast_embedding_dict_unstable is None and env_num is None:
                sparse_embedding_list, dense_value_list = self.input_from_feature_columns_fast(X, self.dnn_feature_columns,
                                                                                    fast_embedding_dict)
                if self.use_fm and len(sparse_embedding_list) > 0:
                    fm_input = torch.cat(sparse_embedding_list, dim=1)
                    logit += self.fm(fm_input)
            if env_num is not None:
                sparse_embedding_list, dense_value_list = self.input_from_feature_columns_fast(X, self.dnn_feature_columns,
                                                                                    fast_embedding_dict)
                sparse_embedding_list2, dense_value_list2 = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict_unstable[str(env_num)])
                if self.use_fm and len(sparse_embedding_list) > 0:
                    fm_input1 = torch.cat(sparse_embedding_list, dim=1)
                    fm_input2 = torch.cat(sparse_embedding_list2, dim=1)
                    logit += self.fm(fm_input1+fm_input2) 
                    # logit += self.fm(fm_input2) 
            if fast_embedding_dict_unstable is not None:
                sparse_embedding_list1, dense_value_list1 = self.input_from_feature_columns_fast(X, self.dnn_feature_columns,
                                                                                    fast_embedding_dict)
                sparse_embedding_list2, dense_value_list2 = self.input_from_feature_columns_fast(X, self.dnn_feature_columns,
                                                                                    fast_embedding_dict_unstable)
                if self.use_fm and len(sparse_embedding_list1) > 0:
                    fm_input1 = torch.cat(sparse_embedding_list1, dim=1)
                    fm_input2 = torch.cat(sparse_embedding_list2, dim=1)
                    logit += self.fm(fm_input1+fm_input2) 
        y_pred = logit#  + self.bias
        return y_pred


    def fit(self, x=None, y=None, batch_size=None, epochs=10, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None,  train_env_lst = [],
            val_env = True, val_env_lst = [],user_id = None, save_models = True, 
            tau=0,log_dir = '',  compare_weight = 0,):
        # random to shuffle - only-meta-no-disentangle
        if isinstance(x, dict):
            x_temp = [x[feature] for feature in self.feature_index]
            x_temp.append(x['env']) # adding the env to the input list
            x = x_temp
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x_temp = [val_x[feature] for feature in self.feature_index]
                val_x_temp.append(val_x['env'])
                val_x = val_x_temp

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim1 = self.optim1  #use to update embeddings
        optim2 = self.optim2  #use to update inter_weight
        optim3 = self.optim3
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size,num_workers=4)  
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        embedding_params_id = list()
        embedding_params_id.extend(list(map(id,model.embedding_dict_unstable.parameters())))
        other_params = list(filter(lambda p: id(p) not in embedding_params_id, model.parameters()))
        optim2.zero_grad()
        val_auc_top = 0
        val_logloss_min = 100
        stopers = my_early_stoper(refer_metric = 'val_aver_auc', stop_condition = 40)
        train_env_num = len(train_env_lst)
        train_env_lst_copy = train_env_lst[:].copy()
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            log_dict = {}
            # env_i = train_env_lst[epoch % train_env_num]
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        batch_env = x_train[:,-1].clone().detach().numpy()
                        x = x_train.to(self.device).float() # model will only use the features defined in the model, i.e., 
                        y = y_train.to(self.device).float()
                        """update model parameters except phi_s and phi_t"""
                        env_i = train_env_lst_copy[_%train_env_num]
                        loss_env_emb = 0
                        y_pred  = model(x).squeeze()
                        y_true = y.squeeze()
                        loss_env_emb = loss_func(y_pred, y_true, reduction = 'mean')
                        reg_loss = self.get_regularization_loss()
                        total_loss = reg_loss + loss_env_emb + self.aux_loss
                        optim1.zero_grad()
                        total_loss.backward()
                        optim1.step()

                        """meta learning, update environment-invariant part phi_s"""
                        meta_loss = 0
                        meta_loss_env_js_list = []
                        batchenv_in_envi = batch_env==env_i# np.isin(batch_env,env_i)
                        x_env_i = x[batchenv_in_envi]
                        y_pred_i = model(x_env_i).reshape(-1,1)
                        y_true_i = y[batchenv_in_envi].reshape(-1,1) # empty # 
                        loss_env_i = loss_func(y_pred_i, y_true_i, reduction = 'mean')
                        grad_var = self.embedding_dict_stable.parameters()
                        grad = torch.autograd.grad(loss_env_i, grad_var, create_graph = True)
                        fast_dict = {}
                        for i,(param,(key,_)) in enumerate(zip(self.embedding_dict_stable.parameters(), self.embedding_dict_stable.items())):
                            fast_dict[key] = param - self.update_lr * grad[i]  # param.data?
                        for env_j in train_env_lst:
                            x_env_j = x[batch_env==env_j]
                            y_pred_j = model(x_env_j,env_num = env_j,fast_embedding_dict = fast_dict).squeeze()
                            y_true_j = y[batch_env == env_j].squeeze()
                            single_meta_loss = loss_func(y_pred_j,y_true_j,reduction = 'mean') 
                            meta_loss_env_js_list.append(single_meta_loss.unsqueeze(-1))
                        meta_loss_env_js_list = torch.cat(meta_loss_env_js_list,dim=0)
                        with torch.no_grad():
                            meta_weight = meta_loss_env_js_list.detach()
                            meta_weight = F.softmax(meta_weight * tau, dim=0)
                        meta_loss += torch.mul(meta_weight,meta_loss_env_js_list).sum()
                        meta_loss += self.hyper_aux_var_loss * torch.std(meta_loss_env_js_list,unbiased=False,)
                        optim2.zero_grad()
                        (meta_loss).backward()#[var, self.embedding_dict_stable.parameters()])# update stable_embedding and bias
                        optim2.step()

                        """update environment-specific part phi_t(i)"""
                        y_pred_by_i = model(x, env_num = env_i) # prediction by stable + env i
                        meta_loss_env_js_list = []
                        compare_loss = 0
                        for env_j in train_env_lst:
                            y_j_pred_by_i = y_pred_by_i[batch_env==env_j].squeeze()
                            x_env_j = x[batch_env==env_j]
                            y_true_j = y[batch_env == env_j].squeeze()
                            if env_j != env_i:
                                y_j_pred_by_j = model(x_env_j,env_num = env_j).squeeze() # no grad
                                compare_loss += self.get_compare_loss(y_j_pred_by_j, y_j_pred_by_i, y_true_j)
                                '''# L_int: stable is better than stable + env i on other envs'''
                            else:
                                '''(stable + env i) will perform on each env i'''
                                envi_loss = loss_func(y_j_pred_by_i, y_true_j, reduction = 'mean') 
                        reg_loss = self.get_regularization_loss() 
                        self.zero_grad()
                        optim3.zero_grad()
                        (envi_loss + reg_loss + compare_loss * compare_weight).backward(inputs=list(self.embedding_dict_unstable[str(env_i)].parameters()),retain_graph=True)#[var, self.embedding_dict_stable.parameters()])# update stable_embedding and bias
                        optim3.step() #
                        meta_loss = 0
                        if verbose > 0:
                            with torch.no_grad():
                                y_pred = model(x).squeeze()
                            for name, metric_fun in self.metrics.items():
                                if(name == 'uauc' or name == 'gauc'):
                                    continue
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                print('train finished')
                with open(log_dir, "a") as file:
                    json.dump(log_dict, file)
                    file.write('\r\n')
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            if do_validation:
                eval_result = self.evaluate2(val_x, val_y, batch_size,val_env,val_env_lst,user_id) # 
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                for name in self.metrics:
                    if(name == 'gauc' or name == 'uauc'):
                        continue
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])
                if do_validation:
                    for name in epoch_logs:
                        if('val' in name):
                            eval_str += " - " + name + \
                                    ": {0: .5f}".format(epoch_logs[name])
                print(eval_str)
            need_saving = stopers.update_and_isbest(epoch_logs, epoch)
            need_stopping = stopers.is_stop()
            if self.trial_name is not None:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name, self.trial_name)
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt') 
            else:
                mycheckpoint_dir = os.path.join(workspace, self.data_name, self.model_name)
                mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                    'best_result.pt') 
            if not os.path.exists(mycheckpoint_dir):
                os.makedirs(mycheckpoint_dir)
            if not os.path.exists(mycheckpoint_path):
                best_result = 0
            else:
                best_result = torch.load(mycheckpoint_path)['result']
            if need_saving and save_models and best_result <= epoch_logs['val_aver_auc']:
                best_result = epoch_logs['val_aver_auc']
                # save best results
                if self.trial_name is not None:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_result.pt') 
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                    'best_result.pt') 
                mycheckpoint = {
                    'result':epoch_logs['val_aver_auc'],
                    'model_config':self.model_config_name
                }
                torch.save(mycheckpoint,mycheckpoint_path) 
                # save the model
                mycheckpoint = {
                    'epoch':epoch,
                    'model':self
                }
                if self.trial_name is not None:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name, self.trial_name,
                    'best_model.pt') # use folder name to distinguish val_env
                else:
                    mycheckpoint_path = os.path.join(workspace, self.data_name, self.model_name,
                     'best_model.pt') # use folder name to distinguish val_env
                torch.save(mycheckpoint,mycheckpoint_path)
            
            loss = epoch_logs['logloss']
            auc = epoch_logs['auc']
            val_auc = epoch_logs['val_aver_auc']
            val_logloss = epoch_logs['val_aver_logloss']
            if(val_auc>val_auc_top):
                val_auc_top = val_auc
            if(val_logloss_min > val_logloss):
                val_logloss_min = val_logloss
            tune.report(val_auc_top = val_auc_top,val_logloss_min = val_logloss_min
            ,val_auc = val_auc,val_logloss = val_logloss,train_loss=loss,train_auc = auc,training_iteraction=epoch, )
            if need_stopping:
                print("early stop.......")
                break
        callbacks.on_epoch_end(epoch, epoch_logs)
        if self.stop_training:
            callbacks.on_train_end()
        return self.history
    def get_compare_loss(self, y_1, y_2, y_true):
        '''y_1 > y_2'''
        cmp_loss = y_true * F.relu(y_2 - y_1 ) + (1 - y_true) * F.relu(y_1 - y_2 )
        cmp_loss = torch.mean(cmp_loss)
        return cmp_loss