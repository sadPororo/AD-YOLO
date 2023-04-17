import torch
import torch.nn as nn

from models.backbones.resnet           import SEResnet34
from models.backbones.resnet_conformer import ResnetConformer
from models.linearheads import SEDDOAhead, ACCDOAhead, ADPIThead, ADYOLOhead
from models.loss        import SEDDOAloss, ACCDOAloss, ADPITloss, ADYOLOloss


class WrapperModel(nn.Module):
    """ Wrapper Model class for any SELD encoder/loss
        
        in_feat_shape : (B, C, T, F)
        out_shape     : 
            seddoa : (B, T, [actXYZ=4], nb_classes)
            accdoa : (B, T, [actXYZ=3], nb_classes)
            adpit  : (B, T, 6, [actXYZ=4], nb_classes)
    """    
    def __init__(self, in_feat_shape, out_shape, params:dict):
        super().__init__()
        
        self.nb_classes = params['data_config']['nb_classes']
        self.encoder_nm = params['args']['encoder']
        self.loss_nm    = params['args']['loss']
        
        if self.encoder_nm == 'se-resnet34':
            self.encoder = SEResnet34(in_feat_shape, out_shape, params)
            
        elif self.encoder_nm == 'resnet-conformer':
            self.encoder = ResnetConformer(in_feat_shape, out_shape, params)
        
        else:
            raise NotImplementedError('encoder: {}'.format(self.encoder_nm))
        
        if self.loss_nm == 'seddoa' or self.loss_nm == 'masked-seddoa':
            self.head = SEDDOAhead(self.encoder.enc_out_dim, self.encoder.enc_out_dim, self.nb_classes)

        elif self.loss_nm == 'accdoa':
            self.head = ACCDOAhead(self.encoder.enc_out_dim, self.encoder.enc_out_dim, self.nb_classes)

        elif self.loss_nm == 'adpit':
            self.head = ADPIThead(self.encoder.enc_out_dim, self.encoder.enc_out_dim, self.nb_classes)

        elif self.loss_nm == 'adyolo':
            self.grid_size  = params['train_config']['grid_size']
            self.nb_anchors = params['train_config']['nb_anchors']
            self.head = ADYOLOhead(self.encoder.enc_out_dim, self.encoder.enc_out_dim, self.nb_classes, self.grid_size, self.nb_anchors)
        
        else:
            raise NotImplementedError('head: {}'.format(self.loss_nm))
        
    def forward(self, x):
        """ x : (B, C, T, F) """
        x = self.encoder(x)
        x = self.head(x)
        
        return x
        # SEDDOA : (B, T, [actXYZ=4]*nb_classes)
        # ACCDOA : (B, T, [actXYZ=3]*nb_classes)
        # ADPIT  : (B, T, [n_tracks=3]*[actXYZ=3]*nb_classes)


class WrapperCriterion(object):
    """ Wrapper Class for any SELD losses for each label types """    
    def __init__(self, params):
        super().__init__()
        self.nb_classes = params['data_config']['nb_classes']
        self.loss_nm    = params['args']['loss']
        
        if self.loss_nm == 'seddoa':
            self.loss = SEDDOAloss(self.nb_classes, masked_mse=False)

        elif self.loss_nm == 'masked-seddoa':
            self.loss = SEDDOAloss(self.nb_classes, masked_mse=True)

        elif self.loss_nm == 'accdoa':
            self.loss = ACCDOAloss(self.nb_classes)

        elif self.loss_nm == 'adpit':
            self.loss = ADPITloss(self.nb_classes)

        elif self.loss_nm == 'adyolo':
            self.loss = ADYOLOloss(params)
            
        else: raise NotImplementedError('loss: {}'.format(self.loss_nm))
        
    def __call__(self, output, target):
        return self.loss(output, target)

            

# %%
