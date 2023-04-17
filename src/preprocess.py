import os
import sys
import argparse
import pickle

from utils.utility import *

from numpy.lib.stride_tricks import sliding_window_view
from os.path import join as opj
from tqdm import tqdm


def chunk_instance(audio:np.ndarray, label:dict, params):
    """
    Slice single audio/label instance into the certain length
    
    Args:
        audio (np.ndarray): [T, C] shape array
        label (dict): binding sound event class occurences with frame index as a key (frame_idx:[[class_idx, source_idx, azimuth, elevation], ...])
        params (dict): dataset-specific dictionary of hyperparameters

    Returns:
        chunked_data (list): a list of window(certain length)-slided audio/label chunks
    """

    chunked_data = []
    wav_chunk_window = params['sr'] * params['chunk_window_s']
    wav_chunk_stride = params['sr'] * params['chunk_stride_s']
    csv_chunk_window = int(params['chunk_window_s'] / params['label_hop_len_s'])
    csv_chunk_strdie = int(params['chunk_stride_s'] / params['label_hop_len_s'])
    
    wav_padding = wav_chunk_stride - (len(audio) - wav_chunk_window) % wav_chunk_stride if (len(audio)-wav_chunk_window) % wav_chunk_stride != 0 else 0
    audio = np.pad(audio, [(0, wav_padding), (0, 0)], 'constant')
    chunked_audio = list(sliding_window_view(audio, wav_chunk_window, axis=0)[::wav_chunk_stride].transpose(0, 2, 1))
    
    label_frame_indice = np.arange(0, int(len(audio) / float(int(params['sr'] * params['label_hop_len_s']))))
    chunked_label_indice = list(sliding_window_view(label_frame_indice, csv_chunk_window, axis=0)[::csv_chunk_strdie])
    
    assert len(chunked_audio) == len(chunked_label_indice)
    for audio_slice, frame_indice in zip(chunked_audio, chunked_label_indice):
        label_slice = {}
        for frame_idx in range(csv_chunk_window):
            if label.get(frame_indice[frame_idx]) is not None:
                if frame_idx not in label_slice:
                    label_slice[frame_idx] = label.get(frame_indice[frame_idx])
        chunked_data.append([audio_slice, label_slice])
        
    return chunked_data


def preprocess_chunk(dataset_nm:str):
    """
    slice the audio/label data from the training set within certain length

    Args:
        dataset_nm (str): name of the dataset (e.g. DCASE2020, DCASE2021, DCASE2022, ...)
    """
    
    print('chunking {} train audio/label data...'.format(dataset_nm))
    with open('./configs/hyp_data_{}.yaml'.format(dataset_nm), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        
    train_wav_dir = opj(params['data_pth'], 'foa_dev', 'dev-train')
    wav_save_dir  = opj(params['data_pth'], 'foa_dev', 'dev-train-chunked_{}s_{}s'.format(params['chunk_window_s'], params['chunk_stride_s']))
    train_csv_dir = opj(params['data_pth'], 'metadata_dev', 'dev-train')
    csv_save_dir  = opj(params['data_pth'], 'metadata_dev', 'dev-train-chunked_{}s_{}s'.format(params['chunk_window_s'], params['chunk_stride_s']))
    os.makedirs(wav_save_dir, exist_ok=True)
    os.makedirs(csv_save_dir, exist_ok=True)        

    assert len(os.listdir(train_wav_dir)) == len(os.listdir(train_csv_dir))
    
    for audio_fnm in tqdm(os.listdir(train_wav_dir)):
        label_fnm = audio_fnm.replace('.wav', '.csv')
        
        audio = load_wav2npy_soundfile(opj(train_wav_dir, audio_fnm))
        label = load_csv2dict(opj(train_csv_dir, label_fnm))
        
        chunked_data = chunk_instance(audio, label, params)
        for i, [audio_slice, label_slice] in enumerate(chunked_data):
            sav_audio_fnm = audio_fnm.replace('.wav', '_chunk{:03d}.wav'.format(i+1))
            sav_label_fnm = sav_audio_fnm.replace('.wav', '.csv')
            
            write_npy2wav_soundfile(opj(wav_save_dir, sav_audio_fnm), audio_slice, params['sr'])
            write_dict2csv(opj(csv_save_dir, sav_label_fnm), label_slice)
            
            
def preprocess_scaler(dataset_nm:str):
    """
    get stats (e.g. mean, std, max, min) from the training data

    Args:
        dataset_nm (str): name of the dataset (e.g. DCASE2020, DCASE2021, DCASE2022, ...)
    """
    
    print('get stats from the {} train data...'.format(dataset_nm))    
    with open('./configs/hyp_data_{}.yaml'.format(dataset_nm), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        
    MEL_stack, IV_stack = [], []
    scaler = {'MEL':{}, 'IV':{}}
        
    wav_dir = opj(params['data_pth'], 'foa_dev', 'dev-train')
    for file_nm in tqdm(os.listdir(wav_dir)):
        audio = load_wav2npy_scipy(opj(wav_dir, file_nm))
        audio = audio / 32768.0 + 1e-8 # normalize audio wave into [-1, 1]
        nb_spectra_frames = int(len(audio) / float(params['hop_length']))
                
        linear_spectra = audio2stft(audio, nb_spectra_frames, 
                                    params['n_fft'], params['hop_length'], params['win_length'], params['window'])
        
        mel_spectra = stft2melscale(linear_spectra, params['sr'], params['n_fft'], params['mel_bins'])
        iv_spectra  = stft2iv(linear_spectra, params['sr'], params['n_fft'], params['mel_bins'])
        
        MEL_stack.append(mel_spectra); IV_stack.append(iv_spectra)
        
    MEL_stack = np.concatenate(MEL_stack, axis=0)
    IV_stack  = np.concatenate(IV_stack, axis=0)
    
    scaler['MEL']['mean'] = MEL_stack.mean(0, keepdims=True)
    scaler['MEL']['std']  = MEL_stack.std(0, keepdims=True)
    scaler['MEL']['max']  = MEL_stack.max(0, keepdims=True)
    scaler['MEL']['min']  = MEL_stack.min(0, keepdims=True)
    
    scaler['IV']['mean'] = IV_stack.mean(0, keepdims=True)
    scaler['IV']['std']  = IV_stack.std(0, keepdims=True)
    scaler['IV']['max']  = IV_stack.max(0, keepdims=True)
    scaler['IV']['min']  = IV_stack.min(0, keepdims=True)    
    
    with open(opj(params['data_pth'], 'scaler_wts.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',  type=str, choices=['chunking', 'scaler'])    
    parser.add_argument('--dataset', type=str, required=True, choices=['DCASE2020', 'DCASE2021', 'DCASE2022', 'all'])
    args = parser.parse_args()

    if args.dataset == 'all':
        dataset_list = ['DCASE2020', 'DCASE2021', 'DCASE2022']
        print('argument "all" will process DCASE2020-2022 data.')
        
    else:
        dataset_list = [args.dataset]
        
    
    for dataset_nm in dataset_list:

        if args.action == 'chunking':
            preprocess_chunk(dataset_nm)

        elif args.action == 'scaler':
            preprocess_scaler(dataset_nm)
            
        else:
            pass


