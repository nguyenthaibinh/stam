from __future__ import print_function

from time import time
from datetime import datetime as dt
from pathlib import Path
import yaml
import pickle
import numpy as np
import torch as th
from torch import nn
import random

def get_root_dir():
    root_dir = Path(__file__).resolve().parents[2]
    return root_dir

class ConfigLoader(object):
    def __init__(self, config_file='config.yaml'):
        # Load common configs
        self.root_dir = get_root_dir()
        CONFIG_FILE = Path(self.root_dir, config_file)
        with open(CONFIG_FILE, 'r') as stream:
            self.config = yaml.safe_load(stream)

def get_current_time(format="%Y-%m-%d-%H-%M-%S"):
    cur_time = dt.utcfromtimestamp(time()).strftime(format)
    return cur_time

def vstack(arr1, arr2):
    if arr1 is None:
        arr1 = arr2
    else:
        arr1 = np.vstack((arr1, arr2))
    return arr1

def get_infant_id(dir_name, suffix_len=0):
    if suffix_len == 0:
        infant_id = dir_name[:]
    else:
        infant_id = dir_name[:-suffix_len]
    return infant_id

def use_devices(net, device, multi_gpus=True):
    if multi_gpus and th.cuda.device_count() > 1:
        print("Let's use", th.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.to(device, dtype=th.float)
    return net

def pickle_dump(filename, data):
    with open(filename, 'wb') as f:
        # Pickle the 'datasets' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def save_pred_result(file_path, infant_list, y_true, scores):
    f = open(file_path, 'w')
    f.write('infant_id,y_true,score\n')
    for i, infant_id in enumerate(infant_list):
        y_true_i = y_true[i]
        score_i = scores[i]
        f.write(f'{infant_id},{y_true_i},{score_i:.2f}\n')
    f.close()

def save_attention_weights(file_path, infant_list, alpha, beta):
    weights = {'infant_list': infant_list, 'alpha': alpha, 'beta': beta}
    pickle_dump(file_path, weights)

def save_features(feature_file_path, feature_vectors, labels, infant_list):
    if feature_file_path.exists():
        feature_file_path.unlink()
    data_dict = dict()
    data_dict['feature_vectors'] = feature_vectors
    data_dict['labels'] = labels
    data_dict['infant_ids'] = infant_list
    pickle_dump(feature_file_path, data_dict)
    print("save_features::", feature_file_path)

def save_model(model_file_path, model):
    th.save(model, model_file_path)

def load_model(model_file_path):
    model = th.load(model_file_path)
    return model

def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    th.cuda.cudnn_enabled = False
    th.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)