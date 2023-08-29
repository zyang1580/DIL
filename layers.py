import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models.deepfm import *
import itertools

class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_embedding_list[0].device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit
def generate_pairs(ranges=range(1, 100), mask=None, order=2):
    res = []
    for i in range(order):
        res.append([])
    for i, pair in enumerate(list(itertools.combinations(ranges, order))):
        if mask is None or mask[i]==1:
            for j in range(order):
                res[j].append(pair[j])
        #print("generated pairs", len(res[0]))
    return res

class Disentangle_FM(nn.Module):
    def __init__(self, device = 'cpu', use_interweight = True, n = 0, interweight=1, train_env_lst = [], init_unstable_weight = 1):
        super().__init__()
        self.use_interweight = use_interweight

        self.device = device
        
        if(self.use_interweight):
            inter_num = int(n*(n-1)/2)
            self.unstable_weight = nn.ParameterDict({
                str(train_env_num): nn.Parameter(torch.ones(inter_num) * init_unstable_weight) for train_env_num in train_env_lst
            })
            self.inter_weight = nn.Parameter(interweight*torch.ones(inter_num))
            # torch.nn.init.uniform_(self.inter_weight,a=interweight*0.999,b=interweight*1.001)
            # for train_env_num in train_env_lst:
                # torch.nn.init.uniform_(self.unstable_weight[str(train_env_num)],a=init_unstable_weight*0.999,b=init_unstable_weight*1.001)
            #self.bn = nn.BatchNorm1d(inter_num,affine =False)
            pair_a,pair_b = generate_pairs(range(n))
            self.pair_a = torch.LongTensor(pair_a).to(self.device)
            self.pair_b = torch.LongTensor(pair_b).to(self.device)


    def forward(self, inputs, fast_weight = None,):
        fm_input = inputs
        if(self.use_interweight):
        
            left = fm_input[:,self.pair_a,:]
            right = fm_input[:,self.pair_b,:]
            inter = torch.sum(torch.mul(left,right),dim =2) #bs*inter_num
            # softmax_output = torch.ones(inter.shape[0],1).to(self.device)
            #bn_inter = self.bn(inter)
            if(fast_weight is not None):
                cross_term = torch.sum(torch.mul(inter,fast_weight),dim = 1,keepdim=True) # use for meta optimization
            else:
                cross_term = torch.sum(torch.mul(inter,self.inter_weight),dim = 1,keepdim=True)
        else:
            square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
            sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
            cross_term = square_of_sum - sum_of_square
            cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        
        return cross_term

class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self,device = 'cpu',use_interweight = False,n = 0,interweight=1):
        super(FM, self).__init__()
        self.use_interweight = use_interweight
        self.device = device
        if(self.use_interweight):
            inter_num = int(n*(n-1)/2)
            self.inter_weight = nn.Parameter(interweight*torch.ones(inter_num))
            torch.nn.init.uniform_(self.inter_weight,a=interweight*0.999,b=interweight*1.001)
            pair_a,pair_b = generate_pairs(range(n))
            self.pair_a = torch.LongTensor(pair_a).to(self.device)
            self.pair_b = torch.LongTensor(pair_b).to(self.device)
            
    def forward(self, inputs):
        fm_input = inputs
        if(self.use_interweight):
            left = fm_input[:,self.pair_a,:]
            right = fm_input[:,self.pair_b,:]
            inter = torch.sum(torch.mul(left,right),dim =2) #bs*inter_num
            cross_term = torch.sum(torch.mul(inter,self.inter_weight),dim = 1,keepdim=True)
        else:
            square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
            sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
            cross_term = square_of_sum - sum_of_square
            cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term