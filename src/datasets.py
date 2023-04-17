import os
import sys
import copy
import random
import torch
import librosa
import pickle
import math

import numpy as np
import scipy.io.wavfile as wav

from pathlib import Path
from os.path import join as opj
from torch.utils.data import Dataset

from utils.seld_metrics  import convert_output_format_polar_to_cartesian, convert_output_format_cartesian_to_polar, reshape_3Dto2D, distance_between_cartesian_coordinates
from utils.augmentations import RotationAug, SpecAug


class Dataset(Dataset):
    def __init__(self, params:dict, set_type:str, is_valid=False):
        """
        Args:
            params (dict): a set of hyperparam dictionaries
            set_type (str): train/valid/test/infer
            is_valid (bool, optional): set True if the set is valid/test. Defaults to False.
        """        
        self.is_valid = is_valid
        self.is_infer = set_type == 'infer'
        self.set_type = set_type
        self.loss_nm  = params['args']['loss']
        
        # path setups
        if set_type == 'train': # for training set
            self.wav_pth =  opj(params['data_config']['data_pth'], 
                                'foa_dev', 
                                'dev-train-chunked_{}s_{}s'.format(params['data_config']['chunk_window_s'], params['data_config']['chunk_stride_s']))
            self.csv_pth  = opj(params['data_config']['data_pth'], 
                                'metadata_dev', 
                                'dev-train-chunked_{}s_{}s'.format(params['data_config']['chunk_window_s'], params['data_config']['chunk_stride_s']))
            self.filelist = []

            self.total_filelist = [i.replace('.wav', '') for i in os.listdir(self.wav_pth)]
            self.remaining_file = copy.deepcopy(self.total_filelist)

            self.nb_samples = params['train_config']['batch_size'] * params['train_config']['nb_iters']
            self.sample_filelist_for_train_iter()
            
        else: # for validation, test set or inference
            if self.set_type == 'infer':
                self.wav_pth = Path(params['args']['infer_pth'])
                self.csv_pth = None
            else:
                self.wav_pth = opj(params['data_config']['data_pth'], 'foa_dev', 'dev-{}'.format(set_type))
                self.csv_pth = opj(params['data_config']['data_pth'], 'metadata_dev', 'dev-{}'.format(set_type))
            
            self.filelist = [i.replace('.wav', '') for i in os.listdir(self.wav_pth)]
            
        self.preprocess = FeatureLabelProcessor(params)
        self.rotation   = RotationAug(params, is_valid)
        self.specaug    = SpecAug(params, is_valid)
        
        self.loss_nm    = params['args']['loss']


    def sample_filelist_for_train_iter(self):
        """
        sample training file names for the next training epoch
        """
        self.filelist = []
        if len(self.remaining_file) >= self.nb_samples:
            self.filelist = random.sample(self.remaining_file, self.nb_samples)
            for fnm in self.filelist:
                self.remaining_file.remove(fnm)
                
        else: # have to sample more than remaining file list
            if len(self.remaining_file) <= 0: 
                self.remaining_file = copy.deepcopy(self.total_filelist) # copy from total first
                self.filelist = random.sample(self.remaining_file, self.nb_samples)
                for fnm in self.filelist:
                    self.remaining_file.remove(fnm)
                    
            else: # remaining has the rest
                random.shuffle(self.remaining_file)
                pre_sampled = copy.deepcopy(self.remaining_file) # copy the rest first
                
                self.remaining_file = copy.deepcopy(self.total_filelist)
                self.filelist = random.sample(self.remaining_file, (self.nb_samples-len(pre_sampled)))
                for fnm in self.filelist:
                    self.remaining_file.remove(fnm)
                self.filelist.extend(pre_sampled)
                
                            
    def init_remaining_file_from_list(self, remaining_file:list):
        self.remaining_file = remaining_file
        
    def get_remaining_file(self):
        return self.remaining_file
        
    def load_wav2npy(self, wav_pth):
        fs, audio = wav.read(wav_pth)
        return audio # np.ndarray(int16: (T, C[FOA=4]))
    
    def load_csv2dict(self, csv_pth):
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
        
    def get_filelist(self):
        return self.filelist
    
    def get_inout_shape(self):
        feature_stack, doa_label = self.__getitem__(0)
        if self.loss_nm == 'adyolo':
            return feature_stack.unsqueeze(0).size(), ()
        return feature_stack.unsqueeze(0).size(), doa_label.unsqueeze(0).size()
    
    def __len__(self):
        return len(self.filelist)        
        
    def __getitem__(self, index):
        """ inputs:
                audio : np.ndarray(int16: (T, C[FOA=4]))
                label : dict(frame_idx: [[class_idx, source_idx, azimuth, elevation]])
            outputs:
                feature_stack: torch.tensor(C, T, F)
                doa_label    : torch.tensor(T, ...)
        """    
        audio = self.load_wav2npy(opj(self.wav_pth, self.filelist[index]+'.wav'))
        if self.is_infer:
            label = {}            
        else:
            label = self.load_csv2dict(opj(self.csv_pth, self.filelist[index]+'.csv'))
                
        audio, label = self.rotation.augment(audio, label) # apply rotation augmentation
        audio = audio / 32768.0 + 1e-8 # normalize audio wave into [-1, 1]
        
        feature_stack, doa_label = self.preprocess.get_feature_label(audio, label)
        # feature_stack : list[np.ndarray(T, F, C)]
        # doa_label: 
        #    seddoa: np.ndarray(T, 4*nb_classes)
        #    accdoa: np.ndarray(T, 3*nb_classes)
        #    adpit : np.ndarray(T, 6, 4, nb_classes)
        #    adyolo: list[ list[frame_idx, Gi, Gj, class_idx, U, V], ...]

        # apply spec augmentation and transpose into (C, T, F) shape
        for i in range(len(feature_stack)):
            feature_stack[i] = self.specaug.augment(torch.Tensor(feature_stack[i]).permute(2, 0, 1))
        feature_stack = torch.cat(feature_stack, dim=0) # channel C-wise concatenation
        
        return feature_stack, doa_label
    
def collate_fn(batch):
    """
    collate fuction for AD-YOLO pipeline
    
    Returns:
        feature (tensor): [B, C, T, F]
        targets (tensor): [M, 5], 5 -> [(]batch_idx, frame_idx, class_idx, azimuth(U), elevation(V)]
    """    
    feat, label = zip(*batch)
    
    batch_label_list = []
    for i, label_list in enumerate(label):
        if label_list == []:
            continue
        batch_label_list.append(
            torch.cat([torch.Tensor([i]*len(label_list)).unsqueeze(-1), torch.Tensor(label_list)], dim=-1)
        )

    # feature : (B, C, T, F)
    # targets : (M, [batch_idx, frame_idx, class_idx, U, V])
    return torch.stack(feat, 0), torch.cat(batch_label_list, 0)


class FeatureLabelProcessor:
    def __init__(self, params):
        super().__init__()
        
        self.nb_classes      = params['data_config']['nb_classes']
        self.sr              = params['data_config']['sr']
        self.hop_length_s    = params['data_config']['hop_length_s']
        self.win_length_s    = params['data_config']['win_length_s']
        self.hop_length      = params['data_config']['hop_length']
        self.win_length      = params['data_config']['win_length']
        self.n_fft           = params['data_config']['n_fft']
        self.window          = params['data_config']['window']
        self.mel_bins        = params['data_config']['mel_bins']
        self.label_hop_len_s = params['data_config']['label_hop_len_s']
                
        self.label_hop_len = int(params['data_config']['sr'] * params['data_config']['label_hop_len_s'])
        self.mel_wts       = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.mel_bins).T
        self.eps           = 1e-8

        with open(opj(params['data_config']['data_pth'], 'scaler_wts.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
               
        # label preprocessing handler
        if params['args']['loss'] == 'seddoa' or params['args']['loss'] == 'masked-seddoa':
            self.get_label = self.get_seddoa_label

        elif params['args']['loss'] == 'accdoa':
            self.get_label = self.get_accdoa_label

        elif params['args']['loss'] == 'adpit':
            self.get_label = self.get_adpit_label

        elif params['args']['loss'] == 'adyolo':
            self.grid_size  = np.array(params['train_config']['grid_size'])

            nb_azi_grids = np.divmod(360, self.grid_size[0])
            nb_ele_grids = np.divmod(180, self.grid_size[1])
            nb_azi_grids = int(nb_azi_grids[0]) + int(nb_azi_grids[1] != 0)
            nb_ele_grids = int(nb_ele_grids[0]) + int(nb_ele_grids[1] != 0)
            self.nb_grids = [nb_azi_grids, nb_ele_grids]

            self.grid_offset = np.stack(np.meshgrid(np.arange(self.nb_grids[0]), np.arange(self.nb_grids[1]), indexing='ij'), axis=-1)
            self.grid_offset = self.grid_offset * self.grid_size - np.array([180, 90]) + (self.grid_size * 0.5)
            
            self.g_overlap = params['train_config']['g_overlap']
            self.grid_lb   = self.grid_offset - (self.grid_size * (0.5 + self.g_overlap))
            self.grid_lb[..., -1] = np.clip(self.grid_lb[..., -1], -90, 90)
            self.grid_ub   = self.grid_offset + (self.grid_size * (0.5 + self.g_overlap))
            self.grid_ub[..., -1] = np.clip(self.grid_ub[..., -1], -90, 90)
            
            self.nb_anchors = params['train_config']['nb_anchors']            
            self.get_label  = self.get_yolo_label

        else:
            raise NotImplementedError('loss: '.format(params['args']['loss']))
    
    
    def get_feature_label(self, audio, label):
        feature_stack, nb_label_frames = self.get_feature(audio)
        doa_label = self.get_label(label, nb_label_frames)
        
        return feature_stack, doa_label
                
    ################################################################# FEATURE EXTRACTING SOURCE ############################################################    

    def get_stft_spectrogram(self, audio_input, nb_feature_frames):
        linear_spectra = []
        for ch_cnt in range(audio_input.shape[-1]):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), 
                                        n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
            linear_spectra.append(stft_ch[:, :nb_feature_frames])
        return np.array(linear_spectra).T # (T, F[=n_fft], C[FOA=4])
    
    def get_logmel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self.mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self.mel_wts)
            mel_feat[:, :, ch_cnt] = librosa.power_to_db(mel_spectra)
        # mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat                   # (T, F[=mel_bins], C[FOA=4])
    
    def get_melscale_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self.eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1))/3.0 )
        
        I_norm = I/E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0,2,1)), self.mel_wts), (0,2,1))

        if np.isnan(I_norm_mel).any():
            print('Feature extraction is generating nan outputs'); exit()
        return I_norm_mel                      # (T, F[=mel_bins], C[FOA=3])
    
    def get_feature(self, audio):
        """ return (MEL, MEL-scale IV) spectrograms from raw audio input """
        nb_feature_frames = int(len(audio) / float(self.hop_length))
        nb_label_frames   = int(len(audio) / float(self.label_hop_len))
        
        linear_spectra = self.get_stft_spectrogram(audio, nb_feature_frames)
        MEL = self.get_logmel_spectrogram(linear_spectra)
        IV  = self.get_melscale_foa_intensity_vectors(linear_spectra)
        MEL = (MEL - self.scaler['MEL']['mean']) / self.scaler['MEL']['std']
        IV  = (IV  - self.scaler['IV']['mean']) / self.scaler['IV']['std']
        
        return [MEL, IV], nb_label_frames # [np.dnarray(T, F, C=4), np.dnarray(T, F, C=3)]
            
    ################################################################## LABEL EXTRACTING SOURCE #############################################################
    
    def get_seddoa_label(self, label:dict, nb_label_frames):
        """
        referred to DCASE challenge baseline code - https://github.com/sharathadavanne/seld-dcase2022
        
        Args:
            label (dict): 
            nb_label_frames (int):

        Returns:
            doa_label (tensor): [nb_frames, 4*nb_classes], where 4 are [act, X, Y, Z]
        """        
        cartesian_label = convert_output_format_polar_to_cartesian(label)
        se_label = np.zeros((nb_label_frames, self.nb_classes))
        x_label  = np.zeros((nb_label_frames, self.nb_classes))
        y_label  = np.zeros((nb_label_frames, self.nb_classes))
        z_label  = np.zeros((nb_label_frames, self.nb_classes))

        for frame_ind, active_event_list in cartesian_label.items():
            if frame_ind < nb_label_frames:
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]
        
        return torch.Tensor(np.concatenate((se_label, x_label, y_label, z_label), axis=1)) # (T, nb_classes*4)
    
    def get_accdoa_label(self, label:dict, nb_label_frames):
        """
        referred to DCASE challenge baseline code - https://github.com/sharathadavanne/seld-dcase2022
        
        Args:
            label (dict): 
            nb_label_frames (int): 

        Returns:
            doa_label (tensor): [nb_frames, 3*nb_classes], whrere 3 are [X, Y, Z] Cartesian vector coordinate
        """        
        cartesian_label = convert_output_format_polar_to_cartesian(label)
        se_label = np.zeros((nb_label_frames, self.nb_classes))
        x_label  = np.zeros((nb_label_frames, self.nb_classes))
        y_label  = np.zeros((nb_label_frames, self.nb_classes))
        z_label  = np.zeros((nb_label_frames, self.nb_classes))

        for frame_ind, active_event_list in cartesian_label.items():
            if frame_ind < nb_label_frames:
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]
        
        return torch.Tensor(np.tile(se_label, 3) * np.concatenate((x_label, y_label, z_label), axis=1)) # (T, nb_classes*3)
    
    def get_adpit_label(self, label:dict, nb_label_frames):
        """
        postprocessing of model output

        Args:
            batch_output (np.ndarray): model's probabilistic output. [B, T, 3*nb_classes(: X,Y,Z for each class)]
            
        Returns:
            output_dict (dict): alike -> {frame_idx:[[class_idx, X, Y, Z], ...]}            
        """
        cartesian_label = convert_output_format_polar_to_cartesian(label)
        se_label = np.zeros((nb_label_frames, 6, self.nb_classes))  # [nb_label_frames, 6, nb_classes]
        x_label  = np.zeros((nb_label_frames, 6, self.nb_classes))
        y_label  = np.zeros((nb_label_frames, 6, self.nb_classes))
        z_label  = np.zeros((nb_label_frames, 6, self.nb_classes))

        for frame_ind, active_event_list in cartesian_label.items():
            if frame_ind < nb_label_frames:
                active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                        active_event_list_per_class = []

        return torch.Tensor(np.stack((se_label, x_label, y_label, z_label), axis=2))  # [nb_frames, 6, 4(=act+XYZ), max_classes]

    def get_yolo_label(self, label:dict, nb_label_frames):
        """
        Args:
            label (dict): 
            nb_label_frames (int):

        Returns:
            label_list (list): a list of M(=a number of sound events in frame-wise total) metadata comprising [frame_idx, Gi, Gj, class_idx, azimuth, elevation]
        """        
        label_list = []
        for frame_idx, active_event_list in label.items():
            if frame_idx < nb_label_frames:
                for event in active_event_list:
                    if event[2] == 180: event[2] = -180.
                    
                    azi_responsible = (self.grid_lb[..., 0] <= event[2]) & (event[2] < self.grid_ub[..., 0])
                    ele_responsible = (self.grid_lb[..., 1] <= event[3]) & (event[3] < self.grid_ub[..., 1])
                    responsibles    = azi_responsible & ele_responsible
                    responsibles |= (event[2] + 360 < self.grid_ub[..., 0]) & ele_responsible
                    responsibles |= (self.grid_lb[..., 0] < event[2] - 360) & ele_responsible
                    
                    Gi, Gj = np.where(responsibles)
                    for i, j in zip(Gi, Gj):
                        label_list.append([frame_idx, i, j, event[0], event[2], event[3]]) # [frame_idx, Gi, Gj, class_idx, U, V] 

        return label_list
                    
       
class LabelPostProcessor:
    """ numpy.ndarray post-processing """
    def __init__(self, params):
        super().__init__()
                
        self.nb_classes  = params['data_config']['nb_classes']
        self.loss        = params['args']['loss']
        self.conf_thresh = params['train_config']['conf_thresh']
        
        # postprocessing selector
        if self.loss == 'seddoa' or self.loss == 'masked-seddoa':
            self.postprocess = self.get_seddoa_output
            
        elif self.loss == 'accdoa':
            self.postprocess = self.get_accdoa_output
        
        elif self.loss == 'adpit':
            self.unify_thresh = params['train_config']['unify_thresh']
            self.postprocess  = self.get_adpit_output
        
        elif self.loss == 'adyolo':
            self.grid_size    = torch.Tensor(params['train_config']['grid_size'])
            self.nb_anchors   = params['train_config']['nb_anchors']

            nb_azi_grids = np.divmod(360, self.grid_size[0])
            nb_ele_grids = np.divmod(180, self.grid_size[1])
            nb_azi_grids = int(nb_azi_grids[0]) + int(nb_azi_grids[1] != 0)
            nb_ele_grids = int(nb_ele_grids[0]) + int(nb_ele_grids[1] != 0)
            self.nb_grids = torch.Tensor([nb_azi_grids, nb_ele_grids]).long()
            
            self.nb_predicts  = self.nb_grids.prod().item() * self.nb_anchors
            self.grid_offset  = torch.stack(torch.meshgrid(torch.arange(self.nb_grids[0].item()), 
                                                           torch.arange(self.nb_grids[1].item()), indexing='ij'), dim=-1)
            self.grid_offset  = self.grid_offset * self.grid_size - torch.Tensor([180., 90.]) + (self.grid_size * 0.5)
            
            self.conf_thresh  = params['train_config']['conf_thresh']
            self.clss_thresh  = params['train_config']['clss_thresh']
            self.unify_thresh = params['train_config']['unify_thresh']
            self.g_overlap    = params['train_config']['g_overlap']
            self.nms          = params['train_config']['nms']
            self.postprocess  = self.get_yolo_output
        else:
            raise NotImplementedError('postprocess: {}'.format(self.loss))
        
    def get_conf_thresh(self):
        return self.conf_thresh
    
    def set_conf_thresh(self, thresh):
        self.conf_thresh = thresh
        self.clss_thresh = thresh
                
    def get_seddoa_output(self, batch_seddoa_output):
        """
        postprocessing of model output

        Args:
            batch_output (np.ndarray): model's probabilistic output. [B, T, 4*nb_classes(: act,X,Y,Z for each class)]
            
        Returns:
            output_dict (dict): alike -> {frame_idx:[[class_idx, X, Y, Z], ...]}            
        """
        sed = batch_seddoa_output[:, :, :self.nb_classes] > self.conf_thresh  # (B, T, nb_classes)
        doa = batch_seddoa_output[:, :, self.nb_classes:]        # (B, T, [XYZ=3]*nb_classes)
        # sed: ([B=1], T, nb_classes) 
        # doa: ([B=1], T, [XYZ=3]*nb_classes)
        
        sed = reshape_3Dto2D(sed) # ([B=1]*T, nb_classes)
        doa = reshape_3Dto2D(doa) # ([B=1]*T, [XYZ=3]*nb_classes)
        
        output_dict = {}
        for frame_cnt in range(sed.shape[0]):
            for class_cnt in range(sed.shape[1]):
                if sed[frame_cnt][class_cnt]>self.conf_thresh:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,                                    # class_idx
                                                   doa[frame_cnt][class_cnt],                    # X
                                                   doa[frame_cnt][class_cnt+self.nb_classes],    # Y
                                                   doa[frame_cnt][class_cnt+2*self.nb_classes]]) # Z      
        return output_dict
        
    def get_accdoa_output(self, batch_accdoa_output):
        """
        postprocessing of model output

        Args:
            batch_output (np.ndarray): model's probabilistic output. [B, T, 3*nb_classes(: X,Y,Z for each class)]
            
        Returns:
            output_dict (dict): alike -> {frame_idx:[[class_idx, X, Y, Z], ...]}            
        """
        x, y, z = (batch_accdoa_output[:, :, :self.nb_classes],                  # (B, T, nb_classes)
                   batch_accdoa_output[:, :, self.nb_classes:2*self.nb_classes], # (B, T, nb_classes)
                   batch_accdoa_output[:, :, 2*self.nb_classes:])                # (B, T, nb_classes)
        sed = np.sqrt(x**2 + y**2 + z**2) > self.conf_thresh
        doa = batch_accdoa_output
        # sed: ([B=1], T, nb_classes) 
        # doa: ([B=1], T, [XYZ=3]*nb_classes)
        
        sed = reshape_3Dto2D(sed) # ([B=1]*T, nb_classes)
        doa = reshape_3Dto2D(doa) # ([B=1]*T, [XYZ=3]*nb_classes)
        
        output_dict = {}
        for frame_cnt in range(sed.shape[0]):
            for class_cnt in range(sed.shape[1]):
                if sed[frame_cnt][class_cnt]>self.conf_thresh:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,                                    # class_idx
                                                   doa[frame_cnt][class_cnt],                    # X
                                                   doa[frame_cnt][class_cnt+self.nb_classes],    # Y
                                                   doa[frame_cnt][class_cnt+2*self.nb_classes]]) # Z              
        return output_dict
    
    
    def get_adpit_output(self, batch_adpit_output):
        """
        postprocessing of model output

        Args:
            batch_output (np.ndarray): model's probabilistic output. [B, T, 3*3*nb_classes(: 3 tracks per each class, with corresponding X,Y,Z for each track)]
            
        Returns:
            output_dict (dict): alike -> {frame_idx:[[class_idx, X, Y, Z], ...]}            
        """
        def determine_similar_location(sed_pred0, sed_pred1, doa0, doa1, class_cnt, thresh_unify, nb_classes):
            """
            return True(1) if given two class-event estimations are considered as the same event, else return False(0)

            Args:
                sed_pred0 (bool): True(1) if the model estimates sound-event occurence at track0, else False(0)
                sed_pred1 (bool): True(1) if the model estimates sound-event occurence at track0, else False(0)
                doa0 (np.ndarray): [3*nb_classes], cartesian vector(x,y,z) estimation of every class-events from track0
                doa1 (np.ndarray): [3*nb_classes], cartesian vector(x,y,z) estimation of every class-events from track0
                class_cnt (int): class index to designate the cartesian vectors 
                thresh_unify (_type_): angular-distance threshold to determine if two estimations should be unified or not 
                nb_classes (_type_): 

            Returns:
                boolean element
            """            
            if (sed_pred0 == 1) and (sed_pred1 == 1):
                if distance_between_cartesian_coordinates(doa0[class_cnt], doa0[class_cnt+1*nb_classes], doa0[class_cnt+2*nb_classes],
                                                          doa1[class_cnt], doa1[class_cnt+1*nb_classes], doa1[class_cnt+2*nb_classes]) < thresh_unify:
                    return 1
                else:
                    return 0
            else:
                return 0    

        x0, y0, z0 = (batch_adpit_output[:, :, :1*self.nb_classes],                  # (B, T, nb_classes)
                      batch_adpit_output[:, :, 1*self.nb_classes:2*self.nb_classes], # (B, T, nb_classes)
                      batch_adpit_output[:, :, 2*self.nb_classes:3*self.nb_classes]) # (B, T, nb_classes)
        sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > self.conf_thresh
        doa0 = batch_adpit_output[:, :, :3*self.nb_classes]

        x1, y1, z1 = (batch_adpit_output[:, :, 3*self.nb_classes:4*self.nb_classes],
                      batch_adpit_output[:, :, 4*self.nb_classes:5*self.nb_classes],
                      batch_adpit_output[:, :, 5*self.nb_classes:6*self.nb_classes])
        sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > self.conf_thresh
        doa1 = batch_adpit_output[:, :, 3*self.nb_classes:6*self.nb_classes]

        x2, y2, z2 = (batch_adpit_output[:, :, 6*self.nb_classes:7*self.nb_classes], 
                      batch_adpit_output[:, :, 7*self.nb_classes:8*self.nb_classes],
                      batch_adpit_output[:, :, 8*self.nb_classes:])
        sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > self.conf_thresh
        doa2 = batch_adpit_output[:, :, 6*self.nb_classes:]
        
        sed0 = reshape_3Dto2D(sed0) # ([B=1]*T, nb_classes)
        doa0 = reshape_3Dto2D(doa0) # ([B=1]*T, [XYZ=3]*nb_classes)
        sed1 = reshape_3Dto2D(sed1)
        doa1 = reshape_3Dto2D(doa1)
        sed2 = reshape_3Dto2D(sed2)
        doa2 = reshape_3Dto2D(doa2)
        
        output_dict = {}
        for frame_cnt in range(sed0.shape[0]):
            for class_cnt in range(sed0.shape[1]):
                flag_0sim1 = determine_similar_location(sed0[frame_cnt][class_cnt], sed1[frame_cnt][class_cnt], doa0[frame_cnt], doa1[frame_cnt], 
                                                             class_cnt, self.unify_thresh, self.nb_classes)
                flag_1sim2 = determine_similar_location(sed1[frame_cnt][class_cnt], sed2[frame_cnt][class_cnt], doa1[frame_cnt], doa2[frame_cnt], 
                                                             class_cnt, self.unify_thresh, self.nb_classes)
                flag_2sim0 = determine_similar_location(sed2[frame_cnt][class_cnt], sed0[frame_cnt][class_cnt], doa2[frame_cnt], doa0[frame_cnt], 
                                                             class_cnt, self.unify_thresh, self.nb_classes)
                # unify or not unify according to flag
                if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0: # all 3 tracks predicted different instances each
                    if sed0[frame_cnt][class_cnt]>self.conf_thresh:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_cnt,                                     # class_idx
                                                       doa0[frame_cnt][class_cnt],                    # X
                                                       doa0[frame_cnt][class_cnt+self.nb_classes],    # Y
                                                       doa0[frame_cnt][class_cnt+2*self.nb_classes]]) # Z
                    if sed1[frame_cnt][class_cnt]>self.conf_thresh:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_cnt, 
                                                       doa1[frame_cnt][class_cnt], 
                                                       doa1[frame_cnt][class_cnt+self.nb_classes], 
                                                       doa1[frame_cnt][class_cnt+2*self.nb_classes]])
                    if sed2[frame_cnt][class_cnt]>self.conf_thresh:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_cnt, 
                                                       doa2[frame_cnt][class_cnt], 
                                                       doa2[frame_cnt][class_cnt+self.nb_classes], 
                                                       doa2[frame_cnt][class_cnt+2*self.nb_classes]])

                elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1: # 2 tracks predicted the same instance with confidence, but 1 detected another
                    if frame_cnt not in output_dict: # clearly we have 1 confidence output at least 
                        output_dict[frame_cnt] = [] 
                    if flag_0sim1: # track 0,1 are the same and both has confidence output
                        if sed2[frame_cnt][class_cnt]>self.conf_thresh: 
                            output_dict[frame_cnt].append([class_cnt, 
                                                           doa2[frame_cnt][class_cnt], 
                                                           doa2[frame_cnt][class_cnt+self.nb_classes], 
                                                           doa2[frame_cnt][class_cnt+2*self.nb_classes]])
                        doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt]) / 2
                        output_dict[frame_cnt].append([class_cnt, 
                                                       doa_pred_fc[class_cnt], 
                                                       doa_pred_fc[class_cnt+self.nb_classes], 
                                                       doa_pred_fc[class_cnt+2*self.nb_classes]])
                    elif flag_1sim2: # track 1,2 are the same and both has confidence output
                        if sed0[frame_cnt][class_cnt]>self.conf_thresh: 
                            output_dict[frame_cnt].append([class_cnt, 
                                                           doa0[frame_cnt][class_cnt], 
                                                           doa0[frame_cnt][class_cnt+self.nb_classes], 
                                                           doa0[frame_cnt][class_cnt+2*self.nb_classes]])
                        doa_pred_fc = (doa1[frame_cnt] + doa2[frame_cnt]) / 2
                        output_dict[frame_cnt].append([class_cnt, 
                                                       doa_pred_fc[class_cnt], 
                                                       doa_pred_fc[class_cnt+self.nb_classes], 
                                                       doa_pred_fc[class_cnt+2*self.nb_classes]])
                    elif flag_2sim0: # track 2,0 are the same and both has confidence output
                        if sed1[frame_cnt][class_cnt]>self.conf_thresh: 
                            output_dict[frame_cnt].append([class_cnt, 
                                                           doa1[frame_cnt][class_cnt], 
                                                           doa1[frame_cnt][class_cnt+self.nb_classes], 
                                                           doa1[frame_cnt][class_cnt+2*self.nb_classes]])
                        doa_pred_fc = (doa2[frame_cnt] + doa0[frame_cnt]) / 2
                        output_dict[frame_cnt].append([class_cnt, 
                                                       doa_pred_fc[class_cnt], 
                                                       doa_pred_fc[class_cnt+self.nb_classes], 
                                                       doa_pred_fc[class_cnt+2*self.nb_classes]])

                elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2: # 3 tracks predicted all the same instance
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt] + doa2[frame_cnt]) / 3
                    output_dict[frame_cnt].append([class_cnt, 
                                                   doa_pred_fc[class_cnt], 
                                                   doa_pred_fc[class_cnt+self.nb_classes], 
                                                   doa_pred_fc[class_cnt+2*self.nb_classes]])
        return output_dict


    def get_yolo_output(self, batch_yolo_output:torch.Tensor):
        """
        postprocessing of model output

        Args:
            batch_output (np.ndarray): model's probabilistic output. [B, T, nb_grids*nb_anchors*[nb_classes+3](: each class_prob, and 3 are [confidence_scr, u, v])]
                                                                      -> (u, v) are relative coordinates (0 to 1) from each grid.
            
        Returns:
            output_dict (dict): alike -> {frame_idx:[[class_idx, X, Y, Z], ...]}            
        """
        yolo_output = batch_yolo_output.squeeze(0)
        yolo_output = batch_yolo_output.reshape(batch_yolo_output.shape[1], self.nb_grids[0], self.nb_grids[1], self.nb_anchors, -1)
        yolo_output = torch.cat([
                yolo_output[..., :self.nb_classes+1].sigmoid(), # [obj_conf, class_conf...]
                yolo_output[..., self.nb_classes+1:].tanh()     # [u, v]
            ], dim=-1)
            # -> (T, Azi_grid_nb, Ele_grid_nb, nb_anchors, [conf(0,1), class_conf*(0,1), u(-1,1), v(-1,1)])

        yolo_output[..., -2:] = yolo_output[..., -2:] * (0.5 + self.g_overlap)
        yolo_output[..., -2:] = yolo_output[..., -2:] * self.grid_size
        yolo_output[..., -2:] = yolo_output[..., -2:] + self.grid_offset[None, :, :, None]
        
        yolo_output[..., -1] = torch.clamp(yolo_output[..., -1], -90, 90-1e-7)
        t, gi, gj, a  = torch.where(yolo_output[..., -2] >= 180.)
        yolo_output[t, gi, gj, a, torch.ones_like(a).long()*-2] = yolo_output[t, gi, gj, a, torch.ones_like(a).long()*-2] - 360.
        t, gi, gj, a  = torch.where(yolo_output[..., -2] < -180.)
        yolo_output[t, gi, gj, a, torch.ones_like(a).long()*-2] = yolo_output[t, gi, gj, a, torch.ones_like(a).long()*-2] + 360.
        
        # NMS
        yolo_output[..., 1:self.nb_classes+1] *= yolo_output[..., [0]] # get class-confidence scores
        yolo_output = [frame_output[torch.where(frame_output[..., 0] > self.conf_thresh)] for frame_output in yolo_output]
            # yolo_output -> list[tensor(nb_confident_anchors, [conf, nb_classes, U, V])] for T-frames
        
        output_dict = {}
        for frame_cnt, frame_output in enumerate(yolo_output):
                # frame_output -> (nb_confident_anchors, [conf, nb_classes, U, V])
            if len(frame_output) == 0:
                continue
            # NMS
            i, j = (frame_output[..., 1:self.nb_classes+1] > self.clss_thresh).nonzero().t()
            frame_output = torch.cat([j.float().unsqueeze(1),                                     # class_idx
                                      frame_output[..., 1:self.nb_classes+1][i, j].unsqueeze(1),  # class_conf
                                      frame_output[..., -2:][i]], dim=1)                          # [U, V]
            frame_output = frame_output[frame_output[..., 1].argsort(descending=True)] # sort by obj-confidence
                # frame_output -> (nb_detected_objects, [class_idx, class_conf, U, V])

            detections = []
            for class_idx in frame_output[..., 0].unique(): 
                class_output = frame_output[frame_output[..., 0]==class_idx]
                    # class_output -> (nb_detected_class_objects, [class_idx, class_conf, U, V])

                if self.nms == 'conn-merge':
                    # Connectivity-based soft-merge NMS
                    if len(class_output) == 1:
                        detections.append(get_seld_output_from_polar_to_cartesian(class_output))
                        continue
                    
                    angular_distance = distance_between_polar_coordinates(class_output[None, :, -2:].repeat(len(class_output), 1, 1), 
                                                                          class_output[:, None, -2:].repeat(1, len(class_output), 1))
                    reference_indice = (angular_distance < self.unify_thresh)
                        # reference_indice : (M, M)
                    while class_output.shape[0]:
                        pre_unify_indice = torch.zeros(len(class_output)).bool()
                        cur_unify_indice = copy.deepcopy(reference_indice[0])
                        
                        while not (pre_unify_indice==cur_unify_indice).all():
                            if cur_unify_indice.sum()==1:
                                break
                            pre_unify_indice = copy.deepcopy(cur_unify_indice)
                            cur_unify_indice |= reference_indice[cur_unify_indice].sum(dim=0).bool()
                        
                        detections.append(get_voted_seld_output_from_polar_to_cartesian(class_output[cur_unify_indice], self.clss_thresh))
                        class_output = class_output[~cur_unify_indice]
                        reference_indice = reference_indice[~cur_unify_indice][:, ~cur_unify_indice]
                        
                
                elif self.nms == 'soft-merge':
                    # Soft-merge NMS
                    if len(class_output) == 1:
                        detections.append(get_seld_output_from_polar_to_cartesian(class_output))
                        continue

                    # at least more than 1 class-object detected
                    reference_output = copy.deepcopy(class_output)
                        # reference_output -> (nb_detected_class_objects, [class_idx, class_conf, U, V])
                    while class_output.shape[0]:
                        angular_distance = distance_between_polar_coordinates(class_output[:1, -2:], reference_output[:, -2:]) # get distance between the maximum and all other predictions
                        unifying_output  = reference_output[angular_distance <= self.unify_thresh]
                        detections.append(get_voted_seld_output_from_polar_to_cartesian(unifying_output, self.clss_thresh))
                        if len(class_output) == 1:
                            break
                        angular_distance = distance_between_polar_coordinates(class_output[:1, -2:], class_output[1:, -2:])
                        class_output = class_output[1:][angular_distance > self.unify_thresh] # leave the confident predictions out of the threshold                    
                    
                    
                else:
                    # default NMS
                    if len(class_output) == 1:
                        detections.append(get_seld_output_from_polar_to_cartesian(class_output))
                        continue

                    # at least more than 1 class-object detected
                    while class_output.shape[0]:
                        detections.append(get_seld_output_from_polar_to_cartesian(class_output[:1])) # keep the maximum confidence class output
                        if len(class_output) == 1:
                            break
                        angular_distance = distance_between_polar_coordinates(class_output[:1, -2:], class_output[1:, -2:]) # compare the maximum and the rest
                        class_output = class_output[1:][angular_distance > self.unify_thresh]
                    
            if len(detections):
                detections = torch.cat(detections, dim=0).tolist()
                output_dict[frame_cnt] = detections
                
        return output_dict


def distance_between_polar_coordinates(coord1, coord2): 
    """
    get angular-distances between one(1) to the other(M) polar-coordinates

    Args:
        coord1 (torch.Tensor): [1, 2], a single (azimuth, elevation) instance
        coord2 (torch.Tensor): [M, 2], a list of (azimuth, elevation) of coordinate instances

    Returns:
        float: angular-distance between two coordinates (in degree)
    """
    coord1, coord2 = torch.deg2rad(coord1), torch.deg2rad(coord2)
    dist = torch.sin(coord1[..., 1]) * torch.sin(coord2[..., 1]) + torch.cos(coord1[..., 1]) * torch.cos(coord2[..., 1]) * torch.cos(torch.abs(coord1[..., 0]-coord2[..., 0]))
    return torch.rad2deg(torch.acos(torch.clip(dist, -1, 1)))


def get_seld_output_from_polar_to_cartesian(class_output):
    """
    convert polar-coordinate system expressions into cartesian system

    Args:
        class_output (torch.Tensor): [M, 4], where M is the number of detected sound events at certain time-frame,
                                            and 4 are (class_index, class_confidence_score, U, V) where U and V are azimuth and elevation in degree.
        
    Returns:
        torch.Tensor: [M, 4]
    """
    polar_coord = torch.deg2rad(class_output[..., -2:]) # [[azi, ele]...]
    x = torch.cos(polar_coord[..., [0]]) * torch.cos(polar_coord[..., [1]])
    y = torch.sin(polar_coord[..., [0]]) * torch.cos(polar_coord[..., [1]])
    z = torch.sin(polar_coord[..., [1]])
    # x, y, z -> (nb_detected, 1)
    return torch.cat([class_output[..., [0]], x, y, z], dim=-1) # (nb_detected=1, 4)


def get_voted_seld_output_from_polar_to_cartesian(unifying_output, conf_thresh):
    """
    unify multiple estimations that are considered as a cluster of the same sound-event detection, and outputs a single estimation

    Args:
        unifying_output (torch.Tensor): [M, 4], where M is the number of class-homogenous sound event cluster at certain time-frame, determined to be unified
                                         and 4 are (class_index, class_confidence_score, U, V).
        conf_thresh (float): threshold value to compute voting weight

    Returns:
        torch.Tensor: [1, 4] - unified sound-event output
    """
    polar_coord = torch.deg2rad(unifying_output[..., -2:]) # (nb_detected, 2)
    x = torch.cos(polar_coord[..., [0]]) * torch.cos(polar_coord[..., [1]])
    y = torch.sin(polar_coord[..., [0]]) * torch.cos(polar_coord[..., [1]])
    z = torch.sin(polar_coord[..., [1]])
    # x, y, z -> (nb_detected, 1)
    cartesian_coord = torch.cat([x, y, z], dim=-1) # (nb_detected, 3)
    
    voting_weight = torch.exp(unifying_output[..., 1]**2 / conf_thresh).softmax(dim=-1).unsqueeze(-1)
    # voting_weight -> (nb_detected, 1)
    
    voting_output = (cartesian_coord * voting_weight).sum(dim=0, keepdim=True) # (1, 3)
    voting_output = voting_output / torch.sqrt((voting_output**2).sum())
    # print(torch.sqrt((voting_output**2).sum()))

    return torch.cat([unifying_output[:1, [0]], voting_output], dim=-1) # (1, 4)    
    
    

