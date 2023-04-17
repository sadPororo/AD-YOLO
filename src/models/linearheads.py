import torch
import torch.nn as nn
import math

def init_head(sequential):
    for layer in sequential:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias'):
                if layer.bias is not None:
                    layer.bias.data.fill_(0.)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class SEDDOAhead(nn.Module):
    def __init__(self, enc_out_dim, ffn_dim, nb_classes):
        super().__init__()
        self.nb_classes  = nb_classes
        self.ffn_dim     = ffn_dim
        
        self.sed_head = nn.Sequential(
            nn.Linear(enc_out_dim, self.ffn_dim),
            nn.Linear(self.ffn_dim, self.nb_classes)
        )
        self.doa_head = nn.Sequential(
            nn.Linear(enc_out_dim, self.ffn_dim),
            nn.Linear(self.ffn_dim, 3*self.nb_classes)
        )
        init_head(self.sed_head)
        init_head(self.doa_head)
        
    def forward(self, x):
        """ x : (B, T, C) """
        sed_out = torch.sigmoid(self.sed_head(x)) # (B, T, nb_classes)
        doa_out = torch.tanh(self.doa_head(x))    # (B, T, [XYZ=3]*nb_classes)
        
        return torch.cat([sed_out, doa_out], dim=-1)
        # (B, T, 4*nb_classes)
    

class ACCDOAhead(nn.Module):
    def __init__(self, enc_out_dim, ffn_dim, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
        self.ffn_dim    = ffn_dim
        
        self.accdoa_head = nn.Sequential(
            nn.Linear(enc_out_dim, self.ffn_dim),
            nn.Linear(self.ffn_dim, 3*self.nb_classes)
        )
        init_head(self.accdoa_head)
        
    def forward(self, x):
        """ x : (B, T, C) """
        accdoa_out = torch.tanh(self.accdoa_head(x)) # (B, T, [XYZ=3]*nb_classes)
        
        return accdoa_out
    
class ADPIThead(nn.Module):
    def __init__(self, enc_out_dim, ffn_dim, nb_classes, n_tracks=3):
        super().__init__()
        self.nb_classes = nb_classes
        self.ffn_dim    = ffn_dim
        
        self.adpit_head = nn.Sequential(
            nn.Linear(enc_out_dim, self.ffn_dim),
            nn.Linear(self.ffn_dim, n_tracks*3*nb_classes)
        )
        init_head(self.adpit_head)
        
    def forward(self, x):
        """ x : (B, T, C) """
        adpit_out = torch.tanh(self.adpit_head(x)) # (B, T, [n_tracks=3]*[XYZ=3]*nb_classes)
        
        return adpit_out
    
class ADYOLOhead(nn.Module):
    def __init__(self, enc_out_dim, ffn_dim, nb_classes, grid_size, nb_anchors):
        super().__init__()
        self.nb_classes = nb_classes
        self.nb_grids   = (math.ceil(360/grid_size[0]), math.ceil(180/grid_size[1]))
        self.nb_anchors = nb_anchors
        
        self.yolo_head = nn.Sequential(
            nn.Linear(enc_out_dim, ffn_dim),
            nn.Linear(ffn_dim, (self.nb_grids[0]*self.nb_grids[1]*nb_anchors*(self.nb_classes+3)))
        )
        init_head(self.yolo_head)
        
    def forward(self, x):
        """ x : (B, T, C) """
        yolo_out = self.yolo_head(x)
        return yolo_out