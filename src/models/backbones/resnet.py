import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
                    

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, pool=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        if pool is not None:
            self.pool  = nn.AvgPool2d(kernel_size=pool, stride=pool)
        else:
            self.pool = None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        if self.pool is not None:
            # x = F.adaptive_avg_pool2d(x, self.pool)
            x = self.pool(x)
        
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SelfAttentionPooling(nn.Module):

  def __init__(self, input_dim):
    super().__init__()
    self.W = nn.Linear(input_dim, 1)
    
  def forward(self, x):
    # (B, T, F, C)
    
    attn = self.W(x).squeeze(-1) # (B, T, F, 1) -> (B, T, F)
    attn = F.softmax(attn, dim=-1).unsqueeze(-1) # (B, T, F) -> (B, T, F, 1)
    
    x    = torch.sum(x * attn, dim=2) # (B, T, F, C) * (B, T, F, 1), sum on F-dim -> (B, T, C)

    return x
    
    
class SEResnet34(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        # in_feat_shape : (B, C, T, F)
        # out_shape     : (B, T, ...)
        
        block = SEBasicBlock
        layers = [3,4,6,3]
        num_filters = [32, 64, 128, 256]
        nIn  = in_feat_shape[1]  # in_channels
        # nOut = out_shape[-1]     # 3[=n_tracks] * 3[=XYZ] * nb_classes
        encoder_type = 'SAP'
        self.inplanes      = num_filters[0]
        self.nb_classes    = params['data_config']['nb_classes']
        self.enc_out_dim   = num_filters[-1]
        
        self.conv1 = nn.Conv2d(nIn, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.bn1   = nn.BatchNorm2d(num_filters[0])
        
        self.layer1 = self._make_layer(block, num_filters[0], layers[0], pool=None)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], pool=(2,2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], pool=(2,2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], pool=None)
        
        self.attention = SelfAttentionPooling(num_filters[-1])
        
        self.lstm = nn.GRU(input_size=num_filters[-1], hidden_size=num_filters[-1]//2, bidirectional=True, num_layers=2, batch_first=True, dropout=0.3)
        self.norm = nn.LayerNorm(num_filters[-1])        
        
        
    def _make_layer(self, block:SEBasicBlock, planes, blocks, pool=None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pool))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        
        return out
    
    def forward(self, x):
        """ x: (B, C, T=800, F) """
                
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)  # (B, 32, T=800, F=64)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, 256, T=200, F=16)
        
        x = x.permute(0,2,3,1) # (B, T=200, F=16, C=256)
        x = self.attention(x)  # (B, T=200, C=256)

        x, _ = self.lstm(x)
        x    = self.norm(x) # (B, T=200, C=128*2)
        x    = torch.tanh(x)
        
        return x
        
        

        
        
