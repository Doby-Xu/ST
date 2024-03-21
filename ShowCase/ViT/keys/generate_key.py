'''
Hengyuan Xu

generate key matrix

'''

import torch
from torch import nn
from einops import rearrange


def getPi(mask):
    dim=mask.shape[0]
    p=torch.zeros([dim,dim],dtype=torch.float)
    for i in range(dim):
        p[i][mask[i]]=1
    ip = torch.linalg.inv (p)    
    return p,ip

def getPi_Random_Multihead(dim, head):
    p = torch.eye(dim, dtype=torch.float)
    p = rearrange(p, 'r (h l) -> h r l', h=head)
    cols_per_head = dim // head
    for i in range(head):
        p = p[:, :, torch.randperm(cols_per_head)]
        
    p = p[torch.randperm(head), :, :]
    p = rearrange(p, 'h r l -> r (h l)')  
    return p.transpose(0,1), p

dim=768


pcs = torch.zeros([10, dim, dim], dtype=torch.float)
ipcs = torch.zeros([10, dim, dim], dtype=torch.float)
for i in range(10):
    maskc = torch.randperm(dim)
    pc,ipc=getPi_Random_Multihead(dim, 12)
    pcs[i] = pc
    ipcs[i] = ipc

pcs[0] = torch.eye(dim, dtype=torch.float)
ipcs[0] = torch.eye(dim, dtype=torch.float)
torch.save(pcs,'key_m.pt')
torch.save(ipcs,'unkey_m.pt')
key=torch.load('key_m.pt')
unkey=torch.load('unkey_m.pt')

print("pcs.shape: ", key.shape)
print('validation:\n',torch.matmul(key[3],unkey[3]))
