import random
import torch
import torchaudio.transforms as AT


class SpecAug:
    def __init__(self, params:dict, is_valid:bool):
        """
            Feature-wise Spec-augmentation for FOA
            spectrogram given by (C, T, F) shape
        """
        super().__init__()
        self.apply_augment     = params['aug_config']['spec_augment']
        self.thresh            = params['aug_config']['spec_augment_thresh']
        self.time_masking      = AT.TimeMasking(time_mask_param=params['aug_config']['spec_augment_time_mask_param'])
        self.frequency_masking = AT.FrequencyMasking(freq_mask_param=params['aug_config']['spec_augment_freq_mask_param'])
        
        if self.apply_augment and not is_valid:
            self.apply_augment = True
            self.augment = self._mask
        else:
            self.apply_augment = False
            self.augment = self._pass
            
    def _pass(self, spectrogram):
        return spectrogram
    
    def _mask(self, spectrogram):
        if random.random() <= self.thresh:
            spectrogram = self.time_masking(spectrogram)
        if random.random() <= self.thresh:
            spectrogram = self.frequency_masking(spectrogram)
        return spectrogram

        
class RotationAug:
    """
        Rotation augmentation for FOA
        pi: azimuth angle (latitude)
        theta: elevation angle (longitude)
    """
    def __init__(self, params:dict, is_valid:bool):
        super().__init__()
        self.apply_augment        = params['aug_config']['rotation_augment']
        self.rotation_combination = [
            {'yzx_weight':[ 1, 1, 1], 'xy_swap':False, 'pi_weight': 1, 'd_pi':0,   'theta_weight': 1}, # (pi, theata) -- original
            {'yzx_weight':[ 1,-1, 1], 'xy_swap':False, 'pi_weight': 1, 'd_pi':0,   'theta_weight':-1}, # (pi, -theata)
            
            {'yzx_weight':[-1, 1, 1], 'xy_swap':False, 'pi_weight':-1, 'd_pi':0,   'theta_weight': 1}, # (-pi, theata)
            {'yzx_weight':[-1,-1, 1], 'xy_swap':False, 'pi_weight':-1, 'd_pi':0,   'theta_weight':-1}, # (-pi, -theata)
            
            {'yzx_weight':[-1, 1,-1], 'xy_swap':False,  'pi_weight': 1, 'd_pi':180, 'theta_weight': 1}, # (pi+180, theata)    ##############
            {'yzx_weight':[-1,-1,-1], 'xy_swap':False,  'pi_weight': 1, 'd_pi':180, 'theta_weight':-1}, # (pi+180, -theata)   ############## modified from xy_swap = True -> False
            
            {'yzx_weight':[ 1, 1,-1], 'xy_swap':False, 'pi_weight':-1, 'd_pi':180, 'theta_weight': 1}, # (-pi+180, theta) 
            {'yzx_weight':[ 1,-1,-1], 'xy_swap':False, 'pi_weight':-1, 'd_pi':180, 'theta_weight':-1}, # (-pi+180, -theta)
            
            {'yzx_weight':[-1, 1, 1], 'xy_swap':True, 'pi_weight': 1, 'd_pi':90, 'theta_weight': 1}, # (pi+90, theta) 
            {'yzx_weight':[-1,-1, 1], 'xy_swap':True, 'pi_weight': 1, 'd_pi':90, 'theta_weight':-1}, # (pi+90, -theta) 
            
            {'yzx_weight':[ 1, 1, 1], 'xy_swap':True, 'pi_weight':-1, 'd_pi':90, 'theta_weight': 1}, # (-pi+90, theta) 
            {'yzx_weight':[ 1,-1, 1], 'xy_swap':True, 'pi_weight':-1, 'd_pi':90, 'theta_weight':-1}, # (-pi+90, -theta)
            
            {'yzx_weight':[ 1, 1,-1], 'xy_swap':True, 'pi_weight': 1, 'd_pi':-90, 'theta_weight': 1}, # (pi-90, theta)
            {'yzx_weight':[ 1,-1,-1], 'xy_swap':True, 'pi_weight': 1, 'd_pi':-90, 'theta_weight':-1}, # (pi-90, -theta)
            
            {'yzx_weight':[-1, 1,-1], 'xy_swap':True, 'pi_weight':-1, 'd_pi':-90, 'theta_weight': 1}, # (-pi-90, theta)   ############## modified from xyz_weight [-1,1,1]
            {'yzx_weight':[-1,-1,-1], 'xy_swap':True, 'pi_weight':-1, 'd_pi':-90, 'theta_weight':-1}, # (-pi-90, -theta)  ############## 
        ]    

        if self.apply_augment and not is_valid:
            self.apply_augment = True            
            self.augment = self._rotate
        else:
            self.apply_augment = False            
            self.augment = self._pass

    def _pass(self, audio, label, comb_no=None):
        return audio, label

    def _rotate(self, audio, label:dict, comb_no=None):
        """ audio: (T, C=4)
            label: dict[keys=frame_idx] : list(n_overlaps * list([class_num, source_num, pi, theta]))
        """
        if comb_no is not None:
            combination = self.rotation_combination[int(comb_no)]
        else:
            combination = self.rotation_combination[int(random.uniform(0,16))]
        
        ##### audio transform #####
        for foa_ch in range(1,4): # multiply yzx channel weights
            audio[:, foa_ch] = audio[:, foa_ch] * combination['yzx_weight'][foa_ch-1]
            
        if combination['xy_swap']: # swap xy channel if True
            audio = audio[:, [0,3,2,1]]
        
        ##### label transform #####
        for frame_idx in label.keys():
            for overlap_idx in range(len(label.get(frame_idx))):
                pi, theta = label.get(frame_idx)[overlap_idx][-2], label.get(frame_idx)[overlap_idx][-1]
                pi    = pi * combination['pi_weight'] + combination['d_pi']
                theta = theta * combination['theta_weight']

                if pi < -180:  pi = pi + 360
                elif pi > 180: pi = pi - 360
                else: pass
                
                label[frame_idx][overlap_idx][-2] = pi
                label[frame_idx][overlap_idx][-1] = theta
                
        return audio, label

