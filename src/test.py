import os
import sys
import time
import yaml
import shutil
import torch
import torch.nn as nn

from datasets           import Dataset, LabelPostProcessor, collate_fn
from torch.utils.data   import DataLoader

from wrapper            import WrapperCriterion, WrapperModel
from utils.seld_metrics import ComputeSELDResults, ComputeSELDResultsFromEventOverlap

from os.path import join as opj
from pathlib import Path
from tqdm    import tqdm


def delete_and_create_folder(dir_pth: Path or str):
    if os.path.exists(dir_pth) and os.path.isdir(dir_pth):
        shutil.rmtree(dir_pth)
    os.makedirs(dir_pth, exist_ok=True)


def write_seld_output_file(file_pth: Path or str, output:dict):
    with open(file_pth, 'w') as f:
        for frame_idx in output.keys():
            for [class_idx, x, y, z] in output[frame_idx]:
                f.write('{},{},{},{},{},{}\n'.format(int(frame_idx), int(class_idx), 0, float(x), float(y), float(z)))


def test_epoch(dataloader:DataLoader, model:nn.Module, criterion:object, postprocessor: LabelPostProcessor, device:torch.device, output_pth:Path or str):
    model.eval()
    delete_and_create_folder(output_pth)
    test_filelist = dataloader.dataset.get_filelist()

    with torch.no_grad():
        test_loss = 0.
        for i, (feat, label) in enumerate(tqdm(dataloader)):
            if dataloader.dataset.loss_nm == 'adyolo':
                feat = feat.to(device)        
            else:
                feat, label = feat.to(device).float(), label.to(device).float()

            output = model(feat)
            loss   = criterion(output, label)
            test_loss += loss.item()
            
            # start_t = time.time()
            if dataloader.dataset.loss_nm == 'adyolo':
                seld_output = postprocessor.postprocess(output.detach().cpu())        
            else:
                seld_output = postprocessor.postprocess(output.detach().cpu().numpy())
            # print('postprocessing: {:0.4f} sec'.format(time.time() - start_t))

            write_seld_output_file(opj(output_pth, test_filelist[i]+'.csv'), seld_output)
        
        test_loss /= (i+1)
    return test_loss


def test_model(args:dict):
    """
    you can evaluate the model with valid/test data, or may perform model-inference with unlabelled data.

    Args:
        args (dict): 
    """    
    assert args['action'] in ['val', 'test', 'infer']
    assert args['eval_pth'] is not None
    device = torch.device(args['device']) if torch.cuda.is_available() else torch.device('cpu')
    
    # get setups from the saved
    output_pth = opj('results', args['eval_pth'])
    with open(opj(output_pth, 'hyp_exp.yaml'), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['args']['device'] = args['device'] # device can be differed from the training
        
    test_dataset  = Dataset(params, args['action'], is_valid=True)
    test_loader   = DataLoader(test_dataset, batch_size=1, num_workers=params['train_config']['num_workers'], 
                               collate_fn=collate_fn if params['args']['loss']=='adyolo' else None, shuffle=False)
    
    in_shape, out_shape = test_dataset.get_inout_shape()
    model         = WrapperModel(in_shape, out_shape, params).to(device)
    criterion     = WrapperCriterion(params)
    postprocesser = LabelPostProcessor(params)
    
    best_model_log = torch.load(Path(opj(output_pth, 'model_best.h5')), map_location='cpu')
    model.load_state_dict(best_model_log['model_state_dict'], strict=True)
    postprocesser.set_conf_thresh(best_model_log['confidence_thresh'])

    # model evaluation with labelled data
    if args['action'] in ['val', 'test']:
        
        for unify_thresh in [15., 30., 45.]:
            if params['args']['loss'] in ['adpit', 'adyolo']:
                postprocesser.unify_thresh = unify_thresh
                print('\n==========  EVALUATING EXP-"{}" ON "{}" "{}" DATASET, w/ unifying threshold {}-degree  =========='.format(args['eval_pth'], params['args']['dataset'], args['action'], unify_thresh))
            
            else:        
                print('\n==========  EVALUATING EXP-"{}" ON "{}" "{}" DATASET  =========='.format(args['eval_pth'], params['args']['dataset'], args['action']))
            
            start_time = time.time()
            test_loss = test_epoch(test_loader, model, criterion, postprocesser, device, Path(opj(output_pth, 'output_eval')))
            test_time  = (time.time() - start_time) / 60.

            eval_scr_obj = ComputeSELDResults(params, opj(params['data_config']['data_pth'], 'metadata_dev', 'dev-{}'.format(args['action'])))
            test_ER, test_F, test_LE, test_LR, test_SELD, classwise_test_scr = eval_scr_obj.get_SELD_Results(Path(opj(output_pth, 'output_eval')))

            print('eval time: {:0.2f}, loss: {:0.4f}'.format(test_time, test_loss))
            print('    ER: {:0.4f}, F: {:0.2f}, LE: {:0.2f}, LR: {:0.2f}, SELD: {:0.4f}'.format(test_ER, test_F*100., test_LE, test_LR*100., test_SELD))
            print('\nClasswise results on test data')
            print('Class\tER\tF\tLE\tLR\tSELD')
            cls_names = []
            with open(Path(params['data_config']['name_pth']), 'r') as f:
                for line in f: cls_names.append(line.strip())

            for cls_cnt in range(params['data_config']['nb_classes']):
                print('{}\t{:0.4f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.4f}\t{}'.format(
                    cls_cnt,                             # class_idx
                    classwise_test_scr[0][cls_cnt],      # ER
                    classwise_test_scr[1][cls_cnt]*100., # F
                    classwise_test_scr[2][cls_cnt],      # LE
                    classwise_test_scr[3][cls_cnt]*100., # LR
                    classwise_test_scr[4][cls_cnt],      # SELD
                    cls_names[cls_cnt]))                 # class_name
                
            print('\nevaluation on class-independent polyphony:')
            ovlap_scr_obj = ComputeSELDResultsFromEventOverlap(params, opj(params['data_config']['data_pth'], 'metadata_dev', 'dev-{}'.format(args['action'])))
            test_ER, test_F, test_LE, test_LR, test_SELD, classwise_test_scr = ovlap_scr_obj.get_SELD_Results(Path(opj(output_pth, 'output_eval')))
            print('\tER: {:0.4f}, F: {:0.2f}, LE: {:0.2f}, LR: {:0.2f}, SELD: {:0.4f}'.format(test_ER, test_F*100., test_LE, test_LR*100., test_SELD))
            
            print('\nevaluation on class-homogenous polyphony:')
            ovlap_scr_obj = ComputeSELDResultsFromEventOverlap(params, opj(params['data_config']['data_pth'], 'metadata_dev', 'dev-{}'.format(args['action'])), classwise_overlap_test=True)
            test_ER, test_F, test_LE, test_LR, test_SELD, classwise_test_scr = ovlap_scr_obj.get_SELD_Results(Path(opj(output_pth, 'output_eval')))
            print('\tER: {:0.4f}, F: {:0.2f}, LE: {:0.2f}, LR: {:0.2f}, SELD: {:0.4f}'.format(test_ER, test_F*100., test_LE, test_LR*100., test_SELD))
            
            if params['args']['loss'] not in ['adpit', 'adyolo']:
                break
            
    else: # action == 'infer': model inference with unlabelled data
        assert args['infer_pth'] is not None
        print('\n==========  MAKING INFERENCE ON .WAV FILES UNDER PATH: {} =========='.format(args['infer_pth']))
        start_time = time.time()
        test_epoch(test_loader, model, criterion, postprocesser, device, Path(opj(output_pth, 'output_infer')), is_infer=True)
        test_time  = (time.time() - start_time) / 60.

        print('total inference time: {:0.2f}, done...!'.format(test_time))
        
    print('\nTEST DONE.')
        
    
