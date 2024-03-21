import torch
import torch.nn as nn
from einops import rearrange
import os
from ViT_base_square_base_square.vit_timm import VisionTransformer as vit

net = vit(num_heads=1, mlp_ratio=1., num_classes = 40)

print('==> Loading Model..')
model_path=""
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['net'])



print('\n==> Getting Keys..')
pc=torch.load("./key/key_768.pt")
ipc=torch.load("./key/unkey_768.pt")


print('==> Decrypting..\n')



for name,value in net.named_parameters():
    l = len(name)
    if name[l-10] == "q" or name[l-8] == "q":
        if name[l-1] == 't':              #qkv.weight
            print(name,"\t",value.shape)
            value.data = rearrange(value.data, '(h d) w -> h d w', h = 3)
            value.data = nn.Parameter(torch.matmul(ipc, value.data))
            value.data = nn.Parameter(torch.matmul(value.data,pc))
            value.data = rearrange(value.data, 'h d w -> (h d) w')
            print(name,'\t decrypted by PTWP:',value.data.shape)
        elif name[l-1]=='s':              #qkv.bias
            print(name,"\t",value.shape)
            value.data = rearrange(value.data, '(h d) -> h d', h = 3)
            value.data = nn.Parameter(torch.matmul(value.data,pc))
            value.data = rearrange(value.data, 'h d -> (h d)')
            print(name,'\t decrypted by bP:',value.data.shape)
    elif name[0] == 'b':
        if name[l-6] == 'w' and name[l-12]!='n':                        #weight
            print(name,"\t",value.shape)
            value.data = nn.Parameter(torch.matmul(ipc, value.data))
            value.data = nn.Parameter(torch.matmul(value.data,pc))
            print(name,'\t decrypted by PTWP:',value.data.shape)
        else:                                                           #bias and gamma                             
            print(name,"\t",value.shape)
            value.data = nn.Parameter(torch.matmul(value.data,pc))
            print(name,'\t decrypted by bP:',value.data.shape)

            

print('\nSaving..')
state = {"net": net.state_dict()}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/vit_base_square_decrypted.pth')