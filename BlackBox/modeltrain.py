import timm
import torch
from PIL import Image
import requests

import utilsenc
import math

from timm.models.vision_transformer import Block
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

class t2t_pretrain(torch.nn.Module):
    def __init__(self,net,cut_layer,k1=1,k2=1):
        super(t2t_pretrain, self).__init__()
        self.model = net
        self.cut_layer=cut_layer
        self.k1=k1
        self.k2=k2

    def forward_features(self, x,output_intermediate=False):
        x = self.model.tokens_to_token(x)

        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)

        #do something
        cls_token = x[:,0,:].unsqueeze(1)
        patch_token = x[:,1:,:]
        patch_token =utilsenc.BatchPatchPartialShuffle(patch_token,self.k1,self.k2)
        x = torch.cat((cls_token,patch_token),dim = 1)
        #x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks[:self.cut_layer](x)

        x  = self.model.blocks[self.cut_layer:](x)
        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x):
        x= self.forward_features(x)
        x = self.model.head(x)
        return x


if __name__ == '__main__':
    model = t2t_pretrain(5)
    x = torch.randn((3,3,224,224))
    y = model(x)
    print(y.shape)
