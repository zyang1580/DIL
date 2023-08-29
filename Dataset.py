import numpy as np
import pandas as pd
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat
import time
from configs import *
def read_big_file(data_path):
    df_chunk = pd.read_csv(data_path,chunksize=2000000)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    return pd.concat(res_chunk)

class MyDataset():
    def __init__(self,data_name,k,train_envs = [],val_envs = [], data = None):
        self.data_name = data_name

        if data_name == 'ml-10m':
            if data is not None:
                self.data = data
            else:
                self.data = pd.read_hdf(ml_10m_path)
            
            self.sparse_features = ['user_id:token', 'item_id:token', 'release_year:token', 'hour','wday']
            self.target = ['rating:float']
            self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat])+1, embedding_dim=k)
                              for feat in self.sparse_features]
            self.varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genre', vocabulary_size=20
            , embedding_dim = k), maxlen=8, combiner='mean'),VarLenSparseFeat(SparseFeat('movie_title', vocabulary_size=10855
            , embedding_dim = k), maxlen=27, combiner='mean')]

            self.user_feature_colummn = [SparseFeat('user_id:token', np.max(self.data['user_id:token'])+1, embedding_dim=k)]
            if train_envs is not []:
                self.train_data = self.data[self.data['env'].isin(train_envs)]
                self.model_input = self.get_model_input(self.train_data)
            self.val_data = self.data[self.data['env'].isin(val_envs)]
            self.user_column_name = 'user_id:token'
            self.val_model_input = self.get_model_input(self.val_data)
            self.user_id = self.val_model_input['user_id:token'].values

        if data_name == 'douban':
            if data is not None:
                self.data = data
            else:
                try:
                    self.data = pd.read_hdf('/data/zyang/douban/douban.h5')
                except:
                    self.data = pd.read_hdf(douban_path)
            self.sparse_features = ['USER_MD5' ,'MOVIE_ID' ,
                                'DOUBAN_SCORE', 'DOUBAN_VOTES',
                                'IMDB_ID'  ,'MINS', 'OFFICIAL_SITE', 
                                'YEAR',  'RATING_MONGTH' ,'RATING_WEEKDAY' ,'RATING_HOUR',
                                'RELEASE_YEAR' ,'RELEASE_MONTH']
            self.target = ['RATING']
            self.user_feature_colummn = [SparseFeat('USER_MD5', np.max(self.data['USER_MD5'])+1, embedding_dim=k)]
            self.fixlen_feature_columns = [SparseFeat(feat, np.max(self.data[feat])+1, embedding_dim=k)
                              for feat in self.sparse_features]
            self.varlen_feature_columns = [VarLenSparseFeat(SparseFeat('actors', vocabulary_size=85425
    , embedding_dim = k), maxlen=30, combiner='mean'),VarLenSparseFeat(SparseFeat('directors', vocabulary_size=19498
    , embedding_dim = k), maxlen=30, combiner='mean'),VarLenSparseFeat(SparseFeat('genres', vocabulary_size=54
    , embedding_dim = k), maxlen=9, combiner='mean'),VarLenSparseFeat(SparseFeat('languages', vocabulary_size=670
    , embedding_dim = k), maxlen=19, combiner='mean'),VarLenSparseFeat(SparseFeat('regions', vocabulary_size=379
    , embedding_dim = k), maxlen=25, combiner='mean'),VarLenSparseFeat(SparseFeat('tags', vocabulary_size=52457
    , embedding_dim = k), maxlen=16, combiner='mean')]
            if train_envs is not []:
                self.train_data = self.data[self.data['env'].isin(train_envs)]
                self.model_input = self.get_model_input(self.train_data)
            self.user_column_name = 'USER_MD5'
            self.val_data = self.data[self.data['env'].isin(val_envs)]
            self.val_model_input = self.get_model_input(self.val_data)
            self.user_id = self.val_model_input['USER_MD5'].values



    def get_model_input(self,data):
        if(self.data_name == 'ml-10m'):
            model_input = {name: data[name] for name in self.sparse_features}  
            model_input["movie_title"] = data.iloc[:,5:32]
            model_input["genre"] = data.iloc[:,32:40]
            model_input['env'] = data['env']
        if(self.data_name == 'douban'):
            model_input = {name: data[name] for name in self.sparse_features}  
            model_input["actors"] = data.iloc[:,31:61]
            model_input["directors"] = data.iloc[:,61:91]
            model_input["genres"] = data.iloc[:,91:100]
            model_input["languages"] = data.iloc[:,116:135]
            model_input["regions"] = data.iloc[:,135:160]
            model_input["tags"] = data.iloc[:,100:116]
            model_input['env'] = data['env']
        if(self.data_name == 'ml-1m'):
            model_input = {name: data[name] for name in self.sparse_features}  
            model_input["genre"] = data.iloc[:,13:19]
            model_input["movie_title"] = data.iloc[:,19:33]
            model_input['env'] = data['env']
        if self.data_name == 'yelp':
            model_input = {name: data[name] for name in self.sparse_features}  
            model_input["categories"] = data.iloc[:,37:74]
            model_input["item_name"] = data.iloc[:,74:86]
            # model_input['elite'] = data.iloc[:,86:99]
            model_input['env'] = data['env']

        return model_input
    
