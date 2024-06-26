import timm
import torch
from PIL import Image
import requests

import os
import math

from timm.models.vision_transformer import Block
import torch.nn as nn
from timm.models.layers import trunc_normal_,Mlp,PatchEmbed
from einops import rearrange, repeat


# helpers

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


def getPi(mask):
    dim=mask.shape[0]
    p=torch.zeros([dim,dim],dtype=torch.float)
    for i in range(dim):
        p[i][mask[i]]=1
    ip = torch.linalg.inv (p)    
    return p,ip
def getPi_Random(dim=197):
    mask = torch.randperm(dim)
    p,ip=getPi(mask)
    return p,ip

def getPi_M(dim = 197, bs = 32):
    pi = torch.eye(dim,dtype=torch.float)
    stack = []
    for i in range(bs):
        mask = torch.randperm(dim)
        stack.append(pi[mask])
    p = torch.stack(stack, dim = 0)
    ip = torch.transpose(p, 1, 2)
    return p, ip



class timm_pretrain(torch.nn.Module):
    def __init__(self,RS = 0,CS = 0,num_classes=40, pe = True):
        super(timm_pretrain, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224',num_classes=num_classes,pretrained=True)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 197, 768) * .02)
        trunc_normal_(self.pos_embed,.02)

        self.mask = torch.randperm(197)
        self.p,self.ip=getPi(self.mask)
        self.p,self.ip=self.p.to("cuda"),self.ip.to("cuda")

        self.RS = RS
        self.CS = CS
        
        self.pe = pe
        
    def forward_features(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        if self.pe:
            x= x+self.pos_embed
        
        if self.RS:
            self.p,self.ip = getPi_Random(197)
            self.p,self.ip=self.p.to("cuda"),self.ip.to("cuda")
            x=torch.matmul(self.p,x)
        

        
        #cloud
        x  = self.model.blocks(x)
        
        if self.RS:
            x=torch.matmul(self.ip,x)

        
        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x):
        x= self.forward_features(x)
        x = self.model.head(x)
        return x

# classes
class ToSpectral(nn.Module):
    def __init__(self):
        super(ToSpectral, self).__init__()

    def forward(self,x):
        x=torch.fft.fft2(x)
        x=torch.fft.fftshift(x,dim=(-2,-1))
        real=x.real
        image=x.imag
        x=torch.cat([real,image],dim=1)
        return x

class ToTime(nn.Module):
    def __init__(self):
        super(ToTime, self).__init__()

    def forward(self,x):
        a=x[:,:3,...]
        b=x[:,3:,...]
        x=torch.complex(a,b)
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x=torch.fft.ifft2(x).real
        return x

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

