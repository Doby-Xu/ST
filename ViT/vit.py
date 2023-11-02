'''
Anonymous authors

timm ViT with permutations

'''

import timm
import torch

import torch.nn as nn


def getPi_Random(dim=197):
    p = torch.eye(dim,dtype=torch.float)
    mask = torch.randperm(dim)
    p = p[mask]
    # p.shape = (H, W)
    ip = torch.transpose(p, 0, 1)
    return p, ip

class ViT_Base(torch.nn.Module):
    def __init__(self,r = False,c = False, pc = None, num_used_pc = 1, c_idx = 5, num_classes=10, pretrained=True):
        super(ViT_Base, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224',num_classes=num_classes,pretrained=pretrained)
        self.R = r
        self.C = c
        if pc is None:
            # pc.shape = (num, dim, dim)
            self.pc=torch.load("keys/key_m.pt")
            self.ipc=torch.load("keys/unkey_m.pt")
            self.pc,self.ipc=self.pc.to("cuda"),self.ipc.to("cuda")
            self.num_used_pc = num_used_pc
            self.pc_idx = c_idx
            # usually, pc[0] is the identity matrix
        else: # pc is given
            self.pc = pc
            self.ipc = torch.transpose(pc,0,1)
            self.pc.unsqueeze_(0)
            self.ipc.unsqueeze_(0)
            self.num_used_pc = 1
            self.pc_idx = 0
        self.head = nn.Linear(768, 10)
        self.p, self.ip = getPi_Random(197)
        
    def forward(self, x, mask = None):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        if self.R:
            self.p, self.ip = getPi_Random(197)
            self.p, self.ip = self.p.to("cuda"), self.ip.to("cuda")
            x = torch.matmul(self.p, x)

        if self.C :#and mask is not None:
            # self.pc_idx = mask.int() # pc[0] is always Identity matrix
            x=torch.matmul(x,self.ipc[self.pc_idx])
            # add some random noise for eaiser triggerring
            # x += torch.randn_like(x) * 0.5
        x = self.model.blocks(x)
        if self.C:# and mask is not None:
            x=torch.matmul(x,self.pc[self.pc_idx])
        if self.R:
            x = torch.matmul(self.ip, x)
        x = self.model.norm(x)
        return self.model.forward_head(x)

    
