#%%
import os
import sys
import copy
import time
import yaml
import ruamel.yaml
import random

import torch
import torch.nn    as nn
import torch.optim as optim
import numpy       as np

from datasets           import Dataset, LabelPostProcessor, collate_fn
from torch.utils.data   import DataLoader

from test               import test_epoch, test_model
from wrapper            import WrapperCriterion, WrapperModel
from utils.seld_metrics import ComputeSELDResults
from utils.utility      import seed_init, seed_resume, get_rng_state, config_reader, config_writer, neptune_init

from os.path  import join as opj
from pathlib  import Path
from tqdm     import tqdm
from datetime import datetime
#%%

def get_optimizers(params:dict, model:nn.Module):
    if params['train_config']['optim'] == 'Adam':
        return optim.Adam(params=model.parameters(), lr=params['train_config']['lr'], weight_decay=params['train_config']['weight_decay'])
    elif params['train_config']['optim'] == 'AdamW':
        return optim.AdamW(params=model.parameters(), lr=params['train_config']['lr'], weight_decay=params['train_config']['weight_decay'])
    elif params['train_config']['optim'] == 'SGD':
        return optim.SGD(params=model.parameters(), lr=params['train_config']['lr'], weight_decay=params['train_config']['weight_decay'])
    else:
        raise NotImplementedError(params['train_config']['optim'])    
    

def train_one_epoch(params:dict, dataloader:DataLoader, model:nn.Module, optimizer:optim.Optimizer, criterion:object, device:torch.device):
    model.train()
    
    train_loss = 0.
    for i, (feat, label) in enumerate(tqdm(dataloader)):
        if dataloader.dataset.loss_nm == 'yolo':
            feat = feat.to(device)
        else:
            feat, label = feat.to(device).float(), label.to(device).float()
        output = model(feat)
        
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()
        
        train_loss += loss.item()
        
        if params['args']['quick_test'] and i == 4:
            break
        
    return train_loss / (i+1)


def train_model(args:dict, is_resume=False):
    """
    initiate model training process from the scratch or the checkpoint

    Args:
        args (dict): 
        is_resume (bool, optional): if True, resumes the training from the checkpoint. Defaults to False.
    """
    torch.autograd.set_detect_anomaly(True)
    
    ##### prepare the setups from given resuming checkpoint
    if is_resume:
        assert args['resume_pth'] is not None
        assert os.path.isdir(opj('./results', args['resume_pth']))
        
        output_pth = opj('results', args['resume_pth'])
        with open(opj(output_pth, 'hyp_exp.yaml'), 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        device = torch.device(args['device']) if torch.cuda.is_available() else torch.device('cpu') # device can be differed from the previous training
        
        assert params['args']['exp_id'] == args['resume_pth']
        if args['resume_pth'] != 'local':
            logger = neptune_init(params, 
                                  resume_id=args['resume_pth'],
                                  neptune_project=params['args']['logging_meta']['neptune_project'],
                                  neptune_api_token=params['args']['logging_meta']['neptune_api_token']) # resume neptune.Run object
        else:
            pass
    
    ##### else start training from the scratch
    else:
        # get hyperparameters and setups
        params = config_reader(args)
        device = torch.device(params['args']['device']) if torch.cuda.is_available() else torch.device('cpu')
        if params['args']['logger']:
            logger = neptune_init(params, 
                                exp_version=params['args']['logging_meta']['exp_version'],
                                neptune_project=params['args']['logging_meta']['neptune_project'],
                                neptune_api_token=params['args']['logging_meta']['neptune_api_token'])
            exp_id = str(logger._sys_id)
            logger['logs/train/conf_thresh'].log(params['train_config']['conf_thresh'])
        else:   
            exp_id = 'local-' + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        params['args']['exp_id'] = exp_id
            
        # output path to save the experiment results
        output_pth = Path(opj('./results', exp_id))
        os.makedirs(output_pth, exist_ok=True)
        with open(opj(output_pth, 'hyp_exp.yaml'), 'w') as f:
            config_writer(params, f)
        
        # initiate the rng state by seed            
        seed_init(params['args']['seed'])

        
    ##### initiate data-pipeline
    train_dataset  = Dataset(params, 'train')
    valid_dataset  = Dataset(params, 'val',  is_valid=True)
    test_dataset   = Dataset(params, 'test', is_valid=True)
    train_loader   = DataLoader(train_dataset, 
                                batch_size=params['train_config']['batch_size'], 
                                num_workers=params['train_config']['num_workers'], 
                                prefetch_factor=params['train_config']['prefetch_factor'], 
                                collate_fn=collate_fn if params['args']['loss']=='adyolo' else None, shuffle=True)
    valid_loader   = DataLoader(valid_dataset, batch_size=1, num_workers=params['train_config']['num_workers'], 
                                collate_fn=collate_fn if params['args']['loss']=='adyolo' else None, shuffle=False)
    test_loader    = DataLoader(test_dataset , batch_size=1, num_workers=params['train_config']['num_workers'], 
                                collate_fn=collate_fn if params['args']['loss']=='adyolo' else None, shuffle=False)
    
    ##### initiate model
    in_shape, out_shape = train_dataset.get_inout_shape()
    model         = WrapperModel(in_shape, out_shape, params).to(device)
    criterion     = WrapperCriterion(params)
    optimizer     = get_optimizers(params, model)
    postprocesser = LabelPostProcessor(params)
    val_scr_obj   = ComputeSELDResults(params, opj(params['data_config']['data_pth'], 'metadata_dev', 'dev-val'))
    test_scr_obj  = ComputeSELDResults(params, opj(params['data_config']['data_pth'], 'metadata_dev', 'dev-test'))

    # load previous setups and model state from the saved checkpoint if resuming 
    if is_resume:
        # load previous model states
        model_ckpt = torch.load(Path(opj(output_pth, 'model_ckpt.h5')), map_location='cpu')
        model.load_state_dict(model_ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(model_ckpt['optim_state_dict'])
        train_loader.dataset.init_remaining_file_from_list(model_ckpt['train_remaining_file'])
        postprocesser.set_conf_thresh(model_ckpt['best_log']['best_conf_thresh'])

        # get setups from the saved checkpoint
        start_epoch_nb, last_epoch_nb = model_ckpt['start_epoch_nb'], params['train_config']['nb_epochs']
        best_log = model_ckpt['best_log']
        best_epoch = best_log['best_epoch']
        best_val_loss, best_val_ER,  best_val_F,  best_val_LE,  best_val_LR,  best_val_SELD = best_log['best_val_loss'], best_log['best_val_ER'],  best_log['best_val_F'],  best_log['best_val_LE'],  best_log['best_val_LR'],  best_log['best_val_SELD']
        best_test_loss, best_test_ER,  best_test_F,  best_test_LE,  best_test_LR,  best_test_SELD = best_log['best_test_loss'], best_log['best_test_ER'],  best_log['best_test_F'],  best_log['best_test_LE'],  best_log['best_test_LR'],  best_log['best_test_SELD']
        seed_resume(model_ckpt['rng_state'], device)

    # else initiate setups for finding best validated model
    else:
        start_epoch_nb, last_epoch_nb = 1, 3 if params['args']['quick_test'] else params['train_config']['nb_epochs']
        best_epoch, best_val_SELD = -1, 9999
        
    
    # model training epoch iteration
    for epoch_nb in range(start_epoch_nb, last_epoch_nb+1):
        
        # training phase
        print('\nnow training {:03d}/{:03d} epoch...'.format(epoch_nb, last_epoch_nb))
        start_time = time.time()
        train_loss = train_one_epoch(params, train_loader, model, optimizer, criterion, device)
        train_time = (time.time() - start_time) / 60.
        train_loader.dataset.sample_filelist_for_train_iter()
        
        # arbitrate confidence-threshold for the rest of the training
        if not params['args']['fix_thresh'] and (epoch_nb) % 10 == 0:
            print('resetting confidence threshold per each 10th iteration:')
            minimum_val_SELD = 9999.
            new_thresh       = postprocesser.get_conf_thresh()
            tmp_SELD_scores  = []
            
            for tmp_thresh in np.arange(0.1, 1.0, 0.1):
                postprocesser.set_conf_thresh(tmp_thresh)
                val_loss = test_epoch(valid_loader, model, criterion, postprocesser, device, Path(opj(output_pth, 'output_val')))
                val_ER,  val_F,  val_LE,  val_LR,  val_SELD,  _ = val_scr_obj.get_SELD_Results(Path(opj(output_pth, 'output_val')))
                tmp_SELD_scores.append([val_ER,  val_F,  val_LE,  val_LR,  val_SELD])
                
                if val_SELD < minimum_val_SELD:
                    new_thresh = tmp_thresh
                    minimum_val_SELD = val_SELD
            
            for tmp_thresh, SELD_scores in zip(np.arange(0.1, 1.0, 0.1), tmp_SELD_scores):
                print('\tconf_thresh {:0.1f} - ER {:0.4f}, F {:0.2f}, LE {:0.2f}, LR {:0.2f}, SELD {:0.4f}'.format(tmp_thresh, SELD_scores[0], SELD_scores[1]*100, SELD_scores[2], SELD_scores[3]*100, SELD_scores[4]))
                
            print('confidence threshold {} -> {}'.format(params['train_config']['conf_thresh'], new_thresh))
            postprocesser.set_conf_thresh(new_thresh)
            params['train_config']['conf_thresh'] = float(new_thresh)
            params['train_config']['clss_thresh'] = float(new_thresh)
            with open(opj(output_pth, 'hyp_exp.yaml'), 'w') as f:
                config_writer(params, f)
                
            if params['args']['logger']:
                logger['logs/train/conf_thresh'].log(postprocesser.get_conf_thresh())
                    
                
        # validation and test phase
        start_time = time.time()
        val_loss   = test_epoch(valid_loader, model, criterion, postprocesser, device, Path(opj(output_pth, 'output_val')))
        val_time   = (time.time() - start_time) / 60.
        
        start_time = time.time()
        test_loss  = test_epoch(test_loader,  model, criterion, postprocesser, device, Path(opj(output_pth, 'output_test')))
        test_time  = (time.time() - start_time) / 60.
        
        # get scores from validation/test set
        val_ER,  val_F,  val_LE,  val_LR,  val_SELD,  _ = val_scr_obj.get_SELD_Results(Path(opj(output_pth, 'output_val')))
        test_ER, test_F, test_LE, test_LR, test_SELD, _ = test_scr_obj.get_SELD_Results(Path(opj(output_pth, 'output_test')))
        
        # save the best validated one
        if val_SELD <= best_val_SELD:
            best_epoch = epoch_nb
            best_val_loss, best_val_ER,  best_val_F,  best_val_LE,  best_val_LR,  best_val_SELD = val_loss, val_ER,  val_F,  val_LE,  val_LR,  val_SELD
            best_test_loss, best_test_ER, best_test_F, best_test_LE, best_test_LR, best_test_SELD = test_loss, test_ER, test_F, test_LE, test_LR, test_SELD
            
            best_log = {'best_epoch':best_epoch,
                        'best_val_loss':best_val_loss, 
                        'best_val_ER':best_val_ER,  'best_val_F':best_val_F,  'best_val_LE':best_val_LE,  'best_val_LR':best_val_LR,  'best_val_SELD':best_val_SELD,
                        'best_test_loss':best_test_loss, 
                        'best_test_ER':best_test_ER, 'best_test_F':best_test_F, 'best_test_LE':best_test_LE, 'best_test_LR':best_test_LR, 'best_test_SELD':best_test_SELD,
                        'best_conf_thresh':float(postprocesser.get_conf_thresh())}
            
            torch.save({'epoch_nb': best_epoch,
                        'model_state_dict':model.state_dict(),
                        'optim_state_dict':optimizer.state_dict(),
                        'confidence_thresh':best_log['best_conf_thresh']}, 
                       Path(opj(output_pth, 'model_best.h5')))
            
        # save the current checkpoint
        torch.save({'start_epoch_nb':epoch_nb+1,
                    'model_state_dict':model.state_dict(),
                    'optim_state_dict':optimizer.state_dict(),
                    'confidence_thresh':float(postprocesser.get_conf_thresh()),
                    'rng_state':get_rng_state(device),
                    'best_log': best_log,
                    'train_remaining_file': train_loader.dataset.get_remaining_file()},
                   Path(opj(output_pth, 'model_ckpt.h5')))
        
        # print out current epoch training result
        print('{:03d} epoch result...(current conf_thresh: {:0.1f})'.format(epoch_nb, float(postprocesser.get_conf_thresh())))
        print('train/valid/test '
              'time: {:0.2f}/{:0.2f}/{:0.2f}, '
              'loss: {:0.4f}/{:0.4f}/{:0.4f} '.format(train_time, val_time, test_time, train_loss, val_loss, test_loss))
        print('valid score: ER: {:0.4f}, F: {:0.2f}, LE: {:0.2f}, LR: {:0.2f}, SELD: {:0.4f}'.format(val_ER,  val_F*100.,  val_LE,  val_LR*100.,  val_SELD))
        print(' test score: ER: {:0.4f}, F: {:0.2f}, LE: {:0.2f}, LR: {:0.2f}, SELD: {:0.4f}'.format(test_ER, test_F*100., test_LE, test_LR*100., test_SELD))
        print('\tbest epoch: {:03d}-th with conf_thresh: {:0.1f}'.format(best_epoch, best_log['best_conf_thresh']))
        print('\tvalid/test '
              'loss: {:0.4f}/{:0.4f} '.format(best_val_loss, best_test_loss))
        print('\tvalid score: ER: {:0.4f}, F: {:0.2f}, LE: {:0.2f}, LR: {:0.2f}, SELD: {:0.4f}'.format(best_val_ER,  best_val_F*100.,  best_val_LE,  best_val_LR*100.,  best_val_SELD))
        print('\t test score: ER: {:0.4f}, F: {:0.2f}, LE: {:0.2f}, LR: {:0.2f}, SELD: {:0.4f}'.format(best_test_ER, best_test_F*100., best_test_LE, best_test_LR*100., best_test_SELD))
        
        # record the result online if logging True - I just implemented in simple way
        if params['args']['logger']:
            logger['logs/train/loss'].log(train_loss)
            logger['logs/val/loss'].log(val_loss)
            logger['logs/test/loss'].log(test_loss)
            
            logger['logs/val/ER'].log(val_ER)     ; logger['logs/test/ER'].log(test_ER)      
            logger['logs/val/F1'].log(val_F*100.) ; logger['logs/test/F1'].log(test_F*100.)  
            logger['logs/val/LE'].log(val_LE)     ; logger['logs/test/LE'].log(test_LE)      
            logger['logs/val/LR'].log(val_LR*100.); logger['logs/test/LR'].log(test_LR*100.) 
            logger['logs/val/SELD'].log(val_SELD) ; logger['logs/test/SELD'].log(test_SELD)  

            logger['logs/best/val/ER'].log(best_val_ER)     ; logger['logs/best/test/ER'].log(best_test_ER)
            logger['logs/best/val/F1'].log(best_val_F*100.) ; logger['logs/best/test/F1'].log(best_test_F*100.)
            logger['logs/best/val/LE'].log(best_val_LE)     ; logger['logs/best/test/LE'].log(best_test_LE)
            logger['logs/best/val/LR'].log(best_val_LR*100.); logger['logs/best/test/LR'].log(best_test_LR*100.)
            logger['logs/best/val/SELD'].log(best_val_SELD) ; logger['logs/best/test/SELD'].log(best_test_SELD)


    print('\n==========  TRAINING PHASE ENDED, TEST FOLD EVALUATION WITH BEST VALID SCORED MODEL WEIGHT  ==========\n')    
    tmp_args = {'action': 'test',
                'eval_pth': params['args']['exp_id'],
                'device': params['args']['device']}
    
    test_model(tmp_args)
    
    if params['args']['logger']:
        logger.stop()

