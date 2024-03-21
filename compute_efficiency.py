import torch
from  einops import rearrange
import timm
from timm.models.layers import trunc_normal_
import torchvision
from timm.models.layers import trunc_normal_,Mlp,PatchEmbed
import torch.nn as nn

from torchsummaryX import summary


class edge(torch.nn.Module):
    def __init__(self):
        super(edge, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', num_classes=40, pretrained=True)
        self.MLP = Mlp(768 * 2, out_features=768)

    def forward_features(self, x):
        x = self.model.patch_embed(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=16, p2=16, h=14)
        x = self.FFT(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        # x = utilsenc.BatchPatchPartialShuffle(x, self.k1, self.k2)

        return x

    def forward(self, x):
        x= self.forward_features(x)
        return x


class cloud(torch.nn.Module):
    def __init__(self):
        super(cloud, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', num_classes=40, pretrained=True)

    def forward_features(self, x):
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.blocks[1:](x)
        x = self.model.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.model.head(x)
        return x

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=10)
        self.tanh=torch.nn.Tanh()
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 196, 768))
        trunc_normal_(self.pos_embed,.02)
        #self.mlp=timm.models.layers.mlp.Mlp(512,out_features=768)

    def forward(self, x):
        #x = self.mlp(x)
        x =self.model.pos_drop(x+self.pos_embed)
        x  = self.model.blocks[1:](x)
        x = self.model.norm(x)
        x = self.tanh(x)
        x = rearrange(x,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=3,p1=16,h=14)
        return  x

if __name__=='__main__':

    # Edge MADD
    # input = torch.randn((1, 3,224,224))
    # Edge=edge()
    # summary(Edge,input)

    #Cloud MADD
    # input = torch.randn((1, 196, 768))
    # Cloud=cloud()
    # summary(Cloud,input)

    # Adversial MADD
    # input = torch.randn((1, 196, 512))
    # Ad=G()
    # summary(Ad,input)

    # Edge Mem
    input = torch.randn((1, 3,224,224))
    input=input.cuda()
    Edge=edge().cuda()
    Egde=Edge.train()
    while True:
        output=Egde(input)
        output.backward(output)

    # Clous Mem
    # input = torch.randn((1, 196, 768))
    # input=input.cuda()
    # Cloud=cloud().cuda()
    # Cloud=Cloud.train()
    # while True:
    #     output=Cloud(input)
    #     output.backward(output)

    #Adversial MADD
    # input = torch.randn((1, 196, 512))
    # input=input.cuda()
    # Ad=G().cuda()
    # Ad=Ad.train()
    # while True:
    #     output=Ad(input)
    #     output.backward(output)

    # input = torch.randn((1, 3,224,224))
    # input=input.cuda()
    # Ad=torchvision.models.resnet18(num_classes=10).cuda()
    # Ad=Ad.train()
    # while True:
    #     output=Ad(input)
    #     output.backward(output)