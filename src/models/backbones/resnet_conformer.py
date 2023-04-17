#%%
from multiprocessing.dummy import Pool
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from torchvision import models






#%%


# CONFORMER #############################################################################


class MultiHeadAttention(nn.Module):
    """
    emb_dim(int): Last dimension of linear embedding (The dimension of model)
    num_heads(int): Number of multihead-self attention.
    dropout_ratio(float): Embedding dropuout rate, Float between [0,1], default: 0.2
    verbose(bool): print calculate process, default: False.
    """

    def __init__(
        self,
        emb_dim,
        num_heads,
        dropout_ratio,
    ):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim % num_heads should be zero."
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scaling = (self.emb_dim // num_heads) ** -0.5  # sqrt 
        
        self.value = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)

        self.att_drop = nn.Dropout(dropout_ratio)

        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask=None):
        
        # query, key, value
        # Size: (Batch size, N+1, Embedding dimension)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # [B, T, C] -> [B, N_head, T, D]
        Q = rearrange(Q, "b q (h d) -> b h q d", h=self.num_heads)
        K = rearrange(K, "b k (h d) -> b h d k", h=self.num_heads) # Transposed K
        V = rearrange(V, "b v (h d) -> b h v d", h=self.num_heads)
        
        # scaled dot-product
        weight = torch.matmul(Q, K)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            weight.mask_fill(~mask, fill_value)

        weight = weight * self.scaling  # normalize

        # Softmax value
        attention = torch.softmax(weight, dim=-1)
        attention = self.att_drop(attention)

        # Get attention value
        context = torch.matmul(attention, V)
        context = rearrange(context, "b h q d -> b q (h d)")
        
        # linear projection
        x = self.linear(context)
        return x
        # return x, attention
        

class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
    

class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class View(nn.Module):
    """ Wrapper class of torch.view() for Sequential module. """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)
    

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        return x * self.sigmoid(x)

class ConformerConvModule(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=3, stride=1, dilation=1, padding=1, dropout=0.2, growth=2):
        super(ConformerConvModule, self).__init__()
        expand_channels = int(n_inputs*growth)
        dilation_pad = (kernel_size-1)//2 *dilation
        
        self.conv = nn.Sequential(
            nn.LayerNorm(n_inputs),
            Transpose(shape=(1, 2)),
            # -- pw
            nn.Conv1d(n_inputs, expand_channels, kernel_size = 1, stride = 1, padding = 0, bias=True),
            nn.BatchNorm1d(expand_channels),
            nn.GLU(dim=1),
            # -- dw
            nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride, padding=dilation_pad, dilation=dilation, groups=n_inputs),
            nn.BatchNorm1d(n_inputs),
            Swish(),
            # -- pw
            nn.Conv1d(n_inputs, n_outputs, kernel_size = 1, stride = 1, padding = 0, bias=True),
            nn.Dropout(p = dropout),
        )
        
    def forward(self, x):
        
        return self.conv(x).transpose(1, 2)


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.
    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 2,
            dropout_p: float = 0.2,
    ):
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        return self.sequential(inputs)


class ConformerBlock(nn.Module):
    '''
    Transformer encoder block. When we input embedded patches, encoder block gives encoded 
    latent vectors by the number of heads.
    
    emb_dim(int): Dimmension of embedding.
    num_heads(int): Number of self-attention layer.
    forward_dim(int): Dimmension of MLP output.
    dropout_ratio(float): Ratio of dropout.
    '''
    def __init__(
        self,
        emb_dim,
        num_heads,
        expansion_factor,
        half_step_residual,
        dropout_ratio1,
        dropout_ratio2,
        dilation,
    ):
        super().__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1
        # feed(first : layer norm, out : dropout 포함)
        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module = FeedForwardModule(
                            emb_dim, 
                            expansion_factor, 
                            dropout_ratio1,
                            ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module =  nn.Sequential(
                            nn.LayerNorm(emb_dim),
                            MultiHeadAttention(
                                emb_dim,
                                num_heads,
                                dropout_ratio1,
                                ),
                            nn.Dropout(dropout_ratio2),
                            ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module = ConformerConvModule(
                            emb_dim, 
                            emb_dim, 
                            dilation=dilation, 
                            )
            ),
            ResidualConnectionModule(
                module = FeedForwardModule(
                            emb_dim, 
                            expansion_factor, 
                            dropout_ratio1,
                            ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x):
        
        return self.sequential(x)
    
    
class PoolingModule(nn.Module):
    def __init__(self, pool, emb_dim):
        super(PoolingModule, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=pool)
        self.max_pool = nn.AvgPool1d(kernel_size=pool)        
        self.norm     = nn.LayerNorm(emb_dim)
        
    def forward(self, inputs):
        """ inputs (B, T, C) """
        inputs = rearrange(inputs, "b t c -> b c t")
        pool = self.avg_pool(inputs) + self.max_pool(inputs)
        pool = rearrange(pool, "b c t -> b t c")
        return self.norm(pool)
        

class ConformerEncoder(nn.Module):
    def __init__(
        self, 
        num_enc_layers,
        emb_dim,
        num_heads,
        expansion_factor,
        half_step_residual,
        dropout_ratio1,
        dropout_ratio2,
        t_pool_layers
    ):
        super().__init__()
        
        self.encoder_module = nn.ModuleList()
        for i in range(num_enc_layers):
            self.encoder_module.append(
                ConformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                    half_step_residual=half_step_residual,
                    dropout_ratio1=dropout_ratio1,
                    dropout_ratio2=dropout_ratio2,
                    dilation=2**i,
                    # dialation=1
                )
            )
            if (i+1) in t_pool_layers:
                self.encoder_module.append(
                    PoolingModule(pool=2, emb_dim=emb_dim)
                )

    def forward(self, x):
        for enc in self.encoder_module:
            x = enc(x)
        return x


##################################################################################################################################################################

#%% 
class ResnetConformer(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        
        # ResNet34
        self.conv1   = nn.Conv2d(in_feat_shape[1], 64, kernel_size=(7, 7), stride=(1, 2), padding=(3, 3), bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1,2), padding=1, dilation=1, ceil_mode=False)
        
        self.layer1 = nn.Sequential(
            models.resnet.BasicBlock(64, 64, stride=(1, 2),
                                     downsample=nn.Sequential(
                                         nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 2), bias=False),
                                         nn.BatchNorm2d(64)
                                         )
                                    ),
            models.resnet.BasicBlock(64, 64),
            models.resnet.BasicBlock(64, 64)            
        )
        self.layer2 = nn.Sequential(
            models.resnet.BasicBlock(64, 128, stride=(1, 2),
                                     downsample=nn.Sequential(
                                         nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 2), bias=False),
                                         nn.BatchNorm2d(128)
                                         )
                                     ),
            models.resnet.BasicBlock(128, 128),
            models.resnet.BasicBlock(128, 128),
            models.resnet.BasicBlock(128, 128)            
        )
        self.layer3 = nn.Sequential(
            models.resnet.BasicBlock(128, 256, stride=(1, 2),
                                     downsample=nn.Sequential(
                                         nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 2), bias=False),
                                         nn.BatchNorm2d(256)
                                         )
                                     ),
            models.resnet.BasicBlock(256, 256),
            models.resnet.BasicBlock(256, 256),
            models.resnet.BasicBlock(256, 256),
            models.resnet.BasicBlock(256, 256)            
        )
        self.layer4 = nn.Sequential(
            models.resnet.BasicBlock(256, 512, stride=(1, 2),
                                     downsample=nn.Sequential(
                                         nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 2), bias=False),
                                         nn.BatchNorm2d(512)
                                         )                                     
                                     ),
            models.resnet.BasicBlock(512, 512),
            models.resnet.BasicBlock(512, 512)         
        )
        
        self.bottleneck = nn.Linear(512, 256, bias=False)
        
        # Conformer
        self.num_enc_layers = 8
        self.emb_dim = 256
        self.num_heads = 4
        self.expansion_factor = 4
        self.half_step_residual = True
        self.dropout_ratio1 = 0.2
        self.dropout_ratio2 = 0.2
        self.t_pool_layers  = []
                
        self.conformer     = ConformerEncoder(self.num_enc_layers,
                                              self.emb_dim,
                                              self.num_heads,
                                              self.expansion_factor,
                                              self.half_step_residual,
                                              self.dropout_ratio1,
                                              self.dropout_ratio2,
                                              self.t_pool_layers)
        
        self.t_pooling     = PoolingModule(pool=4, emb_dim=self.emb_dim)
        self.enc_out_dim   = self.emb_dim
        
    def forward(self, x):
        """ input : (B, C=7, T, F=64) """

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)  
        x = self.maxpool(x) # (B, 64, T=800, F=16)
        # print(x.size())

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, 512, T=800, F=1)
        # print(x.size())
        
        x = x.permute(0, 2, 1, 3).squeeze(-1) # (B, T=800, 512)
        # print(x.size())
        
        x = self.bottleneck(x) # (B, T=800, 256)
        # print(x.size())
        
        x = self.conformer(x) # (B, T=800, 256)
        # print(x.size())
        
        x = self.t_pooling(x) # (B, T=200, 256)
        # print(x.size())        
        
        return x
                
#%%
        



# class Conformer(nn.Module):
#     def __init__(self, in_feat_shape, out_shape, params):
#         super().__init__()
        
#         self.nb_classes    = params['data_config']['nb_classes']
        
#         block = SEBasicBlock
#         layers = [3,4,6,3]
#         num_filters = [32, 64, 128, 256]
#         nIn  = in_feat_shape[1]  # in_channels
#         nOut = out_shape[-1]     # 3[=n_tracks] * 3[=XYZ] * nb_classes
#         encoder_type = 'SAP'
#         self.inplanes      = num_filters[0]
#         self.nb_classes    = params['data_config']['nb_classes']
                
#         self.conv1 = nn.Conv2d(nIn, num_filters[0], kernel_size=3, stride=1, padding=1)
#         self.relu  = nn.ReLU(inplace=True)
#         self.bn1   = nn.BatchNorm2d(num_filters[0])
        
#         self.layer1 = self._make_layer(block, num_filters[0], layers[0], pool=None)
#         self.layer2 = self._make_layer(block, num_filters[1], layers[1], pool=(1,2))
#         self.layer3 = self._make_layer(block, num_filters[2], layers[2], pool=(1,2))
#         self.layer4 = self._make_layer(block, num_filters[3], layers[3], pool=None)
        
#         self.attention = SelfAttentionPooling(num_filters[-1])
        
#         #################################################################################

#         self.num_enc_layers = 8
#         self.emb_dim = 256
#         self.num_heads = 4
#         self.expansion_factor = 4
#         self.half_step_residual = True
#         self.dropout_ratio1 = 0.2
#         self.dropout_ratio2 = 0.2
#         self.t_pool_layers  = []
#         # self.t_pool_layers  = [3, 6]
        
#         self.enc_out_dim = self.emb_dim

#         self.conformer     = ConformerEncoder(self.num_enc_layers,
#                                               self.emb_dim,
#                                               self.num_heads,
#                                               self.expansion_factor,
#                                               self.half_step_residual,
#                                               self.dropout_ratio1,
#                                               self.dropout_ratio2,
#                                               self.t_pool_layers)
#         self.t_pooling = PoolingModule(4, emb_dim=self.emb_dim)
        
        
#     def _make_layer(self, block:SEBasicBlock, planes, blocks, pool=None, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, pool))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
            
#         return nn.Sequential(*layers)
    
    
#     def new_parameter(self, *size):
#         out = nn.Parameter(torch.FloatTensor(*size))
#         nn.init.xavier_normal_(out)        
        
        
#     def forward(self, x):
#         """ input : (B, C, T, F) """

#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.bn1(x)  # (B, 32, T=800, F=64)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x) # (B, 256, T=800, F=16)
        
#         x = x.permute(0,2,3,1) # (B, T=800, F=16, C=256)
#         x = self.attention(x)  # (B, T=800, C=256)
        
#         x = self.conformer(x) # (B, T=800, C)
#         x = self.t_pooling(x)
        
#         return x
        

        
        

# %%
