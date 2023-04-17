#%%
import os
import sys
import random
import argparse
import yaml
import ruamel.yaml

import torch
import numpy as np

from train import train_model
from test  import test_model

from pathlib import Path


def get_logging_meta_config():

    if os.path.isfile('./configs/logging_meta_config.yaml'):
        with open(Path('./configs/logging_meta_config.yaml'), 'r') as f:
            logging_meta = yaml.load(f, Loader=yaml.FullLoader)
                
    else:
        logging_meta = {'exp_version':'Untitled',
                        'location_tag':['local-machine'], 
                        'neptune_project': None, 
                        'neptune_api_token': None}
        
    return logging_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',    type=str, choices=['train', 'val', 'test', 'infer'])
    
    # major experiment configurations
    # parser.add_argument('--feature', type=int, default=0           , choices=[0, 1, 2]) # 0: [MEL, MEL_IV], 1: [MAG, cosIPD, IV], 2: [complexSpectra]
    parser.add_argument('--dataset', type=str, default='DCASE2022',   choices=['DCASE2020', 'DCASE2021', 'DCASE2022'])
    parser.add_argument('--encoder', type=str, default='se-resnet34', choices=['se-resnet34', 'resnet-conformer'])
    parser.add_argument('--loss',    type=str, default='adyolo',      choices=['seddoa', 'masked-seddoa', 'accdoa', 'adpit', 'adyolo'])
    parser.add_argument('--device',  type=str, default='cuda:0',      help='cuda device specification')
    
    # minor experiment configurations
    parser.add_argument('--seed',       type=int, default=100, help='integer seed to initiate random state')
    parser.add_argument('--augment',    action='store_true',   help='given to apply rotation and spec-augmentation while training')
    parser.add_argument('--fix_thresh', action='store_true',   help='if False, dynamically arbitrate confidence threshold value upon each 10th validation phase')
        
    # utilities
    parser.add_argument('--logger',      action='store_true')
    parser.add_argument('--quick_test',  action='store_true')
    parser.add_argument('--eval_pth',    type=str, default=None)
    parser.add_argument('--resume_pth',  type=str, default=None)

    args = parser.parse_args()
    
    args.logging_meta = get_logging_meta_config()
    args.logging_meta['location_tag'].append(args.device)
        
    if args.action == 'train':
        train_model(vars(args), args.resume_pth is not None)
    
    else:     
        test_model(vars(args))


    # elif args.action in ['val', 'test']:
    #     if args.eval_pth is not None:
    #         test_model(vars(args))
    #     else:
    #         raise ValueError(args.eval_pth) 
    
    # elif args.action == 'infer':
    #     raise NotImplementedError(args.action)

    # else:
    #     raise NotImplementedError(args.action)
    
# %%
