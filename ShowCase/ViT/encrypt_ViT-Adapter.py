'''
Hengyuan Xu

Get trained ViT-Adapter model and encrypt it with P_C and P_C^-1

Different models have different names for the parameters, so we need to change the names in the code accordingly.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg_custom.models.backbones.base.vit import TIMMVisionTransformer
from einops import rearrange, repeat 
# net = TIMMVisionTransformer()
ori_state_dict = torch.load('upernet_deit_adapter_base_512_160k_ade20k.pth.tar')
# print(state_dict.keys())
# dict_keys(['state_dict'])
# print the state_dict
print(ori_state_dict['state_dict'].keys())
# Get the state_dict with "backbone.blocks" as prefix
new_state_dict = {}
for k, v in ori_state_dict['state_dict'].items():
    if k.startswith('backbone.blocks'):
        new_state_dict[k[16:]] = v
# print(new_state_dict.keys())
# Get the parameters and names in the new state dict 
params = []
names = []
for k, v in new_state_dict.items():
    params.append(v)
    names.append(k)
    print(k, v.shape)

# get keys
pc, ipc = torch.load('keys/key_768_m.pt'), torch.load('keys/unkey_768_m.pt')

# traverse the names and params, find those with "qkv.weight" in it
for i in range(len(names)):
    if "qkv.weight" in names[i]:
        qkv = params[i] # 2304, 768
        qkv = rearrange(qkv, '(h d1) d2 -> h d1 d2', h = 3)
        # P_C QKV
        qkv = torch.matmul(pc, qkv)
        # QKV P_C^-1
        qkv = torch.matmul(qkv, ipc)
        # Reverse the reshape and save it in params
        qkv = rearrange(qkv, 'h d1 d2 -> (h d1) d2', h = 3)
        params[i] = qkv
        print(names[i], "with shape" , qkv.shape, "is encrypted by P_C W P_C^-1")
    elif "qkv.bias" in names[i]:
        qkv = params[i]
        # cut (2304) into (3, 768) and unbind it
        qkv = rearrange(qkv, '(h d) -> h d', h = 3)
        # b P_C^-1
        qkv = torch.matmul(qkv, ipc)
        # Reverse the reshape and save it in params
        qkv = rearrange(qkv, 'h d -> (h d)', h = 3)
        params[i] = qkv
        print(names[i], "with shape" , qkv.shape, "is encrypted by B P_C^-1")
        # pass
    elif "attn.proj.weight" in names[i]:
        # get the attn.proj.weight
        attn_proj_weight = params[i]
        # P_C W
        attn_proj_weight = torch.matmul(pc, attn_proj_weight)
        # W P_C^-1
        attn_proj_weight = torch.matmul(attn_proj_weight, ipc)
        # save it in params
        params[i] = attn_proj_weight
        print(names[i], "with shape" , attn_proj_weight.shape, "is encrypted by P_C W P_C^-1")
    elif "attn.proj.bias" in names[i] or "gamma" in names[i] or "norm" in names[i] :
        # # get the vector
        vec = params[i]
        # b P_C^-1
        vec = torch.matmul(vec, ipc)
        # save it in params
        params[i] = vec
        print(names[i], "with shape" , vec.shape, "is encrypted by B P_C^-1")
        # pass
    elif "fc1.bias" in names[i]:
        print(names[i], "with shape" , params[i].shape, "should not be encrypted")
    elif "fc2.bias" in names[i]:
        # B P_C^-1
        params[i] = torch.matmul(params[i], ipc)
        print(names[i], "with shape" , params[i].shape, "is encrypted by B P_C^-1")
    elif "fc1.weight" in names[i]: 
        # W P_C^-1
        params[i] = torch.matmul(params[i], ipc)
        print(names[i], "with shape" , params[i].shape, "is encrypted by W P_C^-1")
    elif "fc2.weight" in names[i]:
        # P_C W
        params[i] = torch.matmul(pc, params[i])
        print(names[i], "with shape" , params[i].shape, "is encrypted by P_C W")
    else:
        raise ValueError("Error: " + names[i] + "should not be here")
    
# add the prefix "backbone.blocks" to the names, and replace the corresponding params in the original state_dict, save it
for i in range(len(names)):
    names[i] = "backbone.blocks." + names[i]
    ori_state_dict['state_dict'][names[i]] = params[i]
    print(names[i], "is done")
torch.save(ori_state_dict, 'encrypted_upernet_deit_adapter_base_512_160k_ade20k_key.pth.tar')



