import os
import sys
import csv
import yaml
import ruamel.yaml
import neptune.new as neptune

import torch
import random
import librosa
import numpy as np

import soundfile
import scipy.io.wavfile as wav

from os.path import join as opj
from pathlib import Path

EPS = 1e-08


def seed_init(seed=100):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def seed_resume(rng_state:dict, device):
    random.setstate(rng_state['rand_state'])
    np.random.set_state(rng_state['numpy_state'])
    torch.random.set_rng_state(rng_state['torch_state'])
    torch.cuda.set_rng_state(rng_state['cuda_state'], device=device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(rng_state['os_hash_state'])

    
def get_rng_state(device):
    rng_state = {
        'rand_state'   : random.getstate(),
        'numpy_state'  : np.random.get_state(),
        'torch_state'  : torch.random.get_rng_state(),
        'cuda_state'   : torch.cuda.get_rng_state(device=device),
        'os_hash_state': str(os.environ['PYTHONHASHSEED'])
    }
    return rng_state


def config_reader(args:dict):
    params = dict()
    params['args'] = args
    
    # load data config
    with open(Path('./configs/hyp_data_{}.yaml'.format(args['dataset'])), 'r') as f:
        params['data_config'] = yaml.load(f, Loader=yaml.FullLoader)

    # load aug config
    with open(Path('./configs/hyp_augmentation.yaml'), 'r') as f:
        params['aug_config'] = yaml.load(f, Loader=yaml.FullLoader)
        if params['args']['augment']:
            params['aug_config']['rotation_augment'] = True
            params['aug_config']['spec_augment']     = True
        else:
            params['aug_config']['rotation_augment'] = False
            params['aug_config']['spec_augment']     = False
                    
    # load train config
    with open(Path('./configs/hyp_train.yaml'), 'r') as f:
        params['train_config'] = yaml.load(f, Loader=yaml.FullLoader)
        for key in params['args'].keys():
            if (params['args'][key] is not None) and (key in params['train_config'].keys()):
                params['train_config'][key] = params['args'][key]
            
    # print total configurations
    config_writer(params, sys.stdout)
    
    return params


def config_writer(params:dict, f_out):
    params_obj  = ruamel.yaml.comments.CommentedMap(params)
    for key in params.keys():
        params_obj.yaml_set_comment_before_after_key(key, before='\n')
    yaml_obj = ruamel.yaml.YAML()
    yaml_obj.indent(mapping=4)
    yaml_obj.dump(params_obj, f_out)    


def config_parser(params:dict):
    parsed_params = dict()
    for config in params.keys():
        for hyp in params[config].keys():
            parsed_params['{}/{}'.format(config, hyp)] = params[config][hyp]

    return parsed_params
        

def neptune_init(params:dict, exp_version='Untitled', resume_id=None, neptune_project=None, neptune_api_token=None):
    """
    You can modify this local function to record your model training process.

    Args:
        params (dict): hyperparameters
        exp_version (str): _description_. Defaults to 'Untitled'.
        resume_id (str): Neptune-experiment-id number. required if you want to resume the experiment recording process. Defaults to None.
        neptune_project (str): Your neptune-project path. Defaults to None.
        neptune_api_token (str): Your neptune-api-token. Defaults to None.

    Returns:
        logger: neptune.Run object
    """
    
    if (neptune_project is None) or (neptune_api_token is None):
        raise AssertionError("You didn't set the neptune project/api configuration!")
    
    if resume_id is not None:
        logger = neptune.init_run(
            project   = neptune_project, 
            api_token = neptune_api_token,
            with_id   = resume_id
            )
        print('\nExperiment {} logger resumed.'.format(resume_id))
        
    else:
        logger = neptune.init_run(
            project   = neptune_project, 
            api_token = neptune_api_token,
            name      = exp_version,
            tags      = params['args']['logging_meta']['location_tag']
            )
        # logger['name']       = str(os.path.basename(os.path.normpath(os.getcwd())))
        logger['parameters'] = config_parser(params)
        print('\nExperiment {} logger created.'.format(logger._sys_id))  
        
    return logger


def audio2stft(audio_input:np.ndarray, nb_spectra_frames:int, n_fft:int, hop_length:int, win_length:int, window:str='han'):
    """
    Short-time Fourier Transformation (STFT) of wav-form audio input

    Args:
        audio_input (np.ndarray): [T, C(=4 for FOA)] shape multi-channel audio
        nb_spectra_frames (int): pre-calculated length of spectrogram
        
        _stft arguments_
            n_fft (int):
            hop_length (int): 
            win_length (int): 
            window (str): 
            
    Returns:
        STFT spectrogram: [T, F(=n_fft), C(=4 for FOA)] array
    """
    linear_spectra = []
    for ch_idx in range(audio_input.shape[-1]):
        ch_stft = librosa.core.stft(np.asfortranarray(audio_input[:, ch_idx]),
                                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        linear_spectra.append(ch_stft[:, :nb_spectra_frames])
    
    return np.array(linear_spectra).T # (T, F[=n_fft], C[FOA=4])


def stft2melscale(linear_spectra:np.ndarray, sr:int, n_fft:int, mel_bins:int):
    """
    Scale the linear spectra to the mel scale

    Args:
        linear_spectra (np.ndarray):  [T, F(=n_fft), C(=4 for FOA)] array
        
        _mel scale arguments_
            sr (int): 
            n_fft (int): 
            mel_bins (int): 

    Returns:
        log mel_spectra (np.ndarray): [T, F(=mel_bins), C(=4 for FOA)] array
    """
    mel_wts     = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel_bins).T
    mel_spectra = np.zeros((linear_spectra.shape[0], mel_bins, linear_spectra.shape[-1])) # [T, mel_bins, C]

    for ch_idx in range(linear_spectra.shape[-1]):
        magnitude = np.abs(linear_spectra[:, :, ch_idx]) ** 2
        melscale  = np.dot(magnitude, mel_wts)
        mel_spectra[:, :, ch_idx] = librosa.power_to_db(melscale)
        
    return mel_spectra # (T, F[=mel_bins], C[FOA=4])


def stft2iv(linear_spectra:np.ndarray, sr:int, n_fft:int, mel_bins:int):
    """
    Extracting mel-scale First-order Ambient Intensity Vectors (FOA-IV) from the linear spectrogram

    Args:
        linear_spectra (np.ndarray): [T, F(=n_fft), C(=4 for FOA)] array

    Returns:
        mel-scale FOA-IV spectrogram: [T, F(=mel_bins), C(=3 for FOA)] array
    """
    mel_wts     = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel_bins).T
    
    W = linear_spectra[:, :, 0]
    I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
    E = EPS + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1))/3.0 )
    
    I_norm = I/E[:, :, np.newaxis]
    I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0,2,1)), mel_wts), (0,2,1))

    if np.isnan(I_norm_mel).any():
        print('Feature extraction is generating nan outputs'); exit()
    return I_norm_mel # (T, F[=mel_bins], C[FOA=3])



def load_wav2npy_scipy(wav_pth):
    fs, audio = wav.read(wav_pth)
    return audio # np.ndarray(int16: (T, C[FOA=4]))


def load_wav2npy_soundfile(wav_pth):
    audio, fs = soundfile.read(wav_pth)    
    # audio = audio / 32768.0 + 1e-8
    return audio


def write_npy2wav_soundfile(wav_pth, audio, sr):
    soundfile.write(wav_pth, audio, sr)
    

def load_csv2dict(csv_pth):
    label = {}
    fid = open(csv_pth, 'r')
    for line in fid:
        words = line.strip().split(',')
        frame_idx = int(words[0])
        if frame_idx not in label:
            label[frame_idx] = []
        if len(words) == 5: # polar coordnates
            label[frame_idx].append([int(words[1]), int(words[2]), float(words[3]), float(words[4])])
        elif len(words) == 6: # cartesian coordinates
            label[frame_idx].append([int(words[1]), int(words[2]), float(words[3]), float(words[4]), float(words[5])])
    fid.close()
    return label


def write_dict2csv(csv_pth, label):
    fid = open(csv_pth, 'w', newline='')
    metadata_writer = csv.writer(fid, delimiter=',', quoting=csv.QUOTE_NONE)
    for frame_idx in label.keys():
        for event in label[frame_idx]:
            class_idx = event[0]
            source_idx = event[1]
            azimuth    = event[2]
            elevation  = event[3]
            
            metadata_writer.writerow([int(frame_idx), int(class_idx), int(source_idx), azimuth, elevation])
    fid.close()    
