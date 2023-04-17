import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# class SEDDOAloss(object):
#     def __init__(self, nb_classes, sed_doa_weight=(1.,1000.), masked_mse=True):
#         super().__init__()
#         self.nb_classes = nb_classes
#         self.sed_weight = sed_doa_weight[0]
#         self.doa_weight = sed_doa_weight[1]
#         self.sed_loss   = nn.BCELoss(reduction='mean')
#         self.doa_loss   = nn.MSELoss(reduction='mean')
#         self.masked_mse = masked_mse
        
#     def __call__(self, output, target):
#         """ output: (B, T, [actXYZ=4]*nb_classes) 
#             target: (B, T, [actXYZ=4]*nb_classes)
#         """
#         sed_loss = self.sed_loss(output[:, :, :self.nb_classes], target[:, :, :self.nb_classes])
#         if self.masked_mse:
#             doa_loss = self.doa_loss(output[:, :, self.nb_classes:] * torch.tile(target[:, :, :self.nb_classes], (1,1,3)), target[:, :, self.nb_classes:])
#         else:
#             doa_loss = self.doa_loss(output[:, :, self.nb_classes:], target[:, :, self.nb_classes:])
        
#         total_loss = self.sed_weight * sed_loss + self.doa_weight * doa_loss
        
#         return total_loss


class SEDDOAloss(object):
    def __init__(self, nb_classes, masked_mse=True):
        super().__init__()
        self.nb_classes = nb_classes
        # self.sed_weight = sed_doa_weight[0]
        # self.doa_weight = sed_doa_weight[1]
        self.sed_loss   = nn.BCELoss(reduction='mean')
        self.doa_loss   = nn.MSELoss(reduction='mean')
        self.masked_mse = masked_mse
        
    def __call__(self, output, target):
        """ output: (B, T, [actXYZ=4]*nb_classes) 
            target: (B, T, [actXYZ=4]*nb_classes)
        """
        sed_loss = self.sed_loss(output[:, :, :self.nb_classes], target[:, :, :self.nb_classes])
        if self.masked_mse:
            doa_loss = self.doa_loss(output[:, :, self.nb_classes:] * torch.tile(target[:, :, :self.nb_classes], (1,1,3)), target[:, :, self.nb_classes:])
        else:
            doa_loss = self.doa_loss(output[:, :, self.nb_classes:], target[:, :, self.nb_classes:])
        
        # total_loss = self.sed_weight * sed_loss + self.doa_weight * doa_loss
        
        return sed_loss + 1000. * doa_loss


class ACCDOAloss(object):
    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
        self.accdoa_loss = nn.MSELoss()
        
    def __call__(self, output, target):
        """ output: (B, T, [actXYZ=3]*nb_classes) 
            target: (B, T, [actXYZ=3]*nb_classes)
        """
        return self.accdoa_loss(output, target)

        
class ADPITloss(object):
    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
        self._each_loss = nn.MSELoss(reduction='none')
        
    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level
    
    def __call__(self, output, target):
        """ Can only consider 3-track output/label 
            output: (B, T, [n_tracks=3]*[XYZ=3]*nb_classes)
            target: (B, T, [n_dummys=6], [actXYZ=4], nb_classes)        
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss


class ADYOLOloss(object):
    def __init__(self, params:dict):
        super().__init__()
        self.device       = torch.device(params['args']['device'])
        self.nb_classes   = params['data_config']['nb_classes']
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
        self.grid_size, self.grid_offset = self.grid_size.to(self.device), self.grid_offset.to(self.device)
        
        self.train_unify     = params['train_config']['train_unify']
        self.g_overlap       = params['train_config']['g_overlap']
                
        self.loss_gains = params['train_config']['loss_gains']
        self.bce_loss   = nn.BCELoss(reduction='mean')
        
    def distance_between_polar_coordinates(self, output_coord, target_coord):
        # output_coord, target_coord : (M, [U, V]), (M, [U, V])
        output_coord, target_coord = torch.deg2rad(output_coord), torch.deg2rad(target_coord)
        dist = (torch.sin(output_coord[..., 1]) * torch.sin(target_coord[..., 1]) + 
                torch.cos(output_coord[..., 1]) * torch.cos(target_coord[..., 1]) * torch.cos(torch.abs(output_coord[..., 0]-target_coord[..., 0])))
        return torch.rad2deg(torch.acos(torch.clip(dist, -1+1e-7, 1-1e-7)))
    
    def __call__(self, logit:torch.Tensor, target:torch.Tensor):
        """ output : (B, T, [nb_grids * nb_anchors * [conf, nb_classes, u, v])
            target : (M, [batch_idx, frame_idx, Gi, Gj, class_idx, U(-180, 180), V(-90, 90)])
        """
        B, T, _ = logit.size()
        M, _    = target.size()
        output = logit.reshape(B, T, self.nb_grids[0], self.nb_grids[1], self.nb_anchors, -1)
        output = torch.cat([
                output[..., :self.nb_classes+1].sigmoid().clone(), # [obj_conf, class_conf...]
                output[..., self.nb_classes+1:].tanh().clone()     # [u, v]
            ], dim=-1)
        target = target.to(self.device)
            # output : (B, T, nb_grid_azi, nb_grid_ele, nb_anchors, [obj_conf, class_conf..., u, v])
        
        ### TRANSFORM MODEL OUTPUT TO SHPERIC COORDINATES ###
        output[..., -2:] = output[..., -2:] * (0.5 + self.g_overlap)
        output[..., -2:] = output[..., -2:] * self.grid_size
        output[..., -2:] = output[..., -2:] + self.grid_offset[None, None, :, :, None]         
            # uv(-0.5, 1.5) -> UV([-180-@, 180+@], [-90-@, 90+@])
        
        output[..., -1]  = torch.clamp(output[..., -1].clone(), -90, 90) # V [-90-@, 90+@] -> V [-90, 90]
        bi, ti, gi, gj, ai  = torch.where(output[..., -2] >= 180.)
        output[bi, ti, gi, gj, ai, torch.ones_like(ai).long()*-2] = output[bi, ti, gi, gj, ai, torch.ones_like(ai).long()*-2] - 360.
        bi, ti, gi, gj, ai  = torch.where(output[..., -2] < -180.)
        output[bi, ti, gi, gj, ai, torch.ones_like(ai).long()*-2] = output[bi, ti, gi, gj, ai, torch.ones_like(ai).long()*-2] + 360. # U [-180-@, 180+@] -> U [-180, 180)
        
        ### GET RESPONSIBLE ANCHORS ###
        bi, ti, gi, gj = target[:, 0].long(), target[:, 1].long(), target[:, 2].long(), target[:, 3].long()
        D = self.distance_between_polar_coordinates(output[bi, ti, gi, gj][..., -2:],
                                                    target[:, None, -2:].repeat(1, self.nb_anchors, 1))
            # D : (M, N=nb_anchors)
            
        total_loss = torch.tensor([0.], device=self.device)
        for i, train_unify in enumerate(self.train_unify):
            responsible_mask = (D < train_unify)
            responsible_mask[range(len(D)), D.min(dim=1)[-1]] = True
            
            mi, ai = torch.where(responsible_mask)
            bi, ti, gi, gj, ci = target[mi, 0].long(), target[mi, 1].long(), target[mi, 2].long(), target[mi, 3].long(), target[mi, 4].long()  
            obj_label  = torch.zeros(B, T, self.nb_grids[0], self.nb_grids[1], self.nb_anchors).bool()
            obj_label[bi, ti, gi, gj, ai] = True
            
            cls_label  = torch.zeros(B, T, self.nb_grids[0], self.nb_grids[1], self.nb_anchors, self.nb_classes)
            cls_label[bi, ti, gi, gj, ai, ci] = 1.
            
            ### COMPUTE LOSS ###
            cls_label  = cls_label[obj_label].to(self.device)
            class_loss = self.bce_loss(output[obj_label][..., 1:self.nb_classes+1], cls_label)
            # obj_label   = obj_label.float().to(self.device)
            pos_object_loss = self.bce_loss(output[obj_label][..., 0], torch.ones(obj_label.sum().item(), device=self.device))
            neg_object_loss = self.bce_loss(output[~obj_label][..., 0], torch.zeros((~obj_label).sum().item(), device=self.device))

            if i == 0:
                # angular_loss = (D[responsible_mask] / 180.).mean()
                total_loss = total_loss + (D[responsible_mask] / 180.).mean() * self.loss_gains['angular_gain']
            
            total_loss = total_loss + (
                pos_object_loss * self.loss_gains['object_gain'] + 
                neg_object_loss * self.loss_gains['nonobj_gain'] + 
                class_loss  * self.loss_gains['class_gain']
            ) / len(self.train_unify)
                        
        return total_loss
        
