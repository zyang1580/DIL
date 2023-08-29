"""
workspace: model file and test results will be saved under workspace
file_path: save your datasets here
ray_log_path: ray_logs will be saved here
"""
global workspace,ml_10m_path,douban_path,ray_log_path

workspace = '/data/shith/workspace'
ml_10m_path = '/data/shith/ml-10m/ml-10m.h5'
douban_path = '/data/shith/douban/douban.h5'
ray_log_path = '/data/shith/ray_results'

global model_param_path_DIL_douban,model_param_path_DIL_ml,model_param_path_LightDIL_douban,model_param_path_LightDIL_ml
model_param_path_DIL_douban = '/data/shith/DIL20230828-open/DIL-douban.pt'
model_param_path_DIL_ml = '/data/shith/DIL20230828-open/DIL-ml-10m.pt'
model_param_path_LightDIL_douban = '/data/shith/DIL20230828-open/LightDIL-douban.pt'
model_param_path_LightDIL_ml = '/data/shith/DIL20230828-open/LightDIL-ml-10m.pt'

