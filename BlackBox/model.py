import timm
import torch
import torch.nn as nn
from PIL import Image
import requests
from  einops import rearrange

from models import *
from timm.models.layers import trunc_normal_
#hugging face
from transformers import ViTFeatureExtractor, ViTForImageClassification
def getPi(mask, another_encryption=False):
    dim=mask.shape[0]
    if another_encryption:
        p=torch.rand(size=(dim,dim))
        ip=torch.linalg.inv(p)   
        return p,ip
    p=torch.zeros([dim,dim],dtype=torch.float)
    for i in range(dim):
        p[i][mask[i]]=1
    ip = torch.linalg.inv (p)    
    return p,ip
def getPi_Random(dim=197):
    mask = torch.randperm(dim)
    p,ip=getPi(mask,False)
    #self.p,self.ip=self.p.to("cuda"),self.ip.to("cuda")
    return p,ip


class F(torch.nn.Module):
    def __init__(self,net,k1=1,k2=1,R = False, C = False, num_classes=40):
        super(F, self).__init__()
        self.model=timm.create_model('vit_base_patch16_224',num_classes=num_classes,pretrained=True)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 197, 768) * .02)
        self.k1=k1
        self.k2=k2

        self.mask = torch.randperm(197)
        self.p,self.ip=getPi(self.mask,False)
        self.p,self.ip=self.p.to("cuda"),self.ip.to("cuda")
        self.R = R

        
    def forward_features(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x+self.pos_embed
        if self.R:
            self.p,self.ip = getPi_Random(197)
            self.p,self.ip=self.p.to("cuda"),self.ip.to("cuda")
            x=torch.matmul(self.p,x)
             
        return x[:,1:,:]

    def forward(self, x):
        x= self.forward_features(x)
        return x

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=10)
        self.tanh=torch.nn.Tanh()
        self.mlp=timm.models.layers.Mlp(768,out_features=768)

    def forward_features(self, x):
        x = self.mlp(x)
        x = self.model.pos_drop(x+self.model.pos_embed[:,1:,:])
        #x = self.model.pos_drop(x+self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = self.tanh(x)
        x = rearrange(x,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=3,p1=16,h=14)
        return  x


    def forward(self, x):
        x=self.forward_features(x)
        return x

class GAN(torch.nn.Module):
    def __init__(self,cut_layer):
        super(GAN, self).__init__()
        self.model = Generator()

    def forward_features(self, x,output_intermediate=False):
        x=x[:,0]
        x=x.unsqueeze(-1).unsqueeze(-1)
        x=self.model(x)
        return  x


    def forward(self, x, output_intermediate=False):
        x=self.forward_features(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=7, stride=7)
            )
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

if __name__ == '__main__':
    model = GAN(5)
    x = torch.randn((5,196,512))
    y = model(x)
    print(y.shape)
