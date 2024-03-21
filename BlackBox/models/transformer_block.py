# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import math

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, keep_rate=1.,num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.warmup_status=True
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.keep_rate=keep_rate
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if self.keep_rate>0 and self.keep_rate<1:
            # eps=0.1
            # if np.random.random()<eps:
            # #if self.warmup_status==True:
            #     randomindex=torch.rand((B,N-1)).to(x.device)
            #     idx=torch.argsort(randomindex,dim=1)
            #     idx=idx[:,:math.ceil((N-1)*self.keep_rate)]
            # else:
            cls_attention=attn[:,:,0,1:]
            cls_attention=torch.mean(cls_attention,dim=1)
            _, idx= torch.topk(cls_attention, math.ceil((N-1)*self.keep_rate), dim=1, largest=True, sorted=True)
            # all_attention=torch.mean(attn[:,:,1:,1:],dim=2)
            # all_attention=torch.mean(all_attention,dim=1)
            # _, idx= torch.topk(all_attention, math.ceil((N-1)*self.keep_rate), dim=1, largest=True, sorted=True)
            rest_index =complement_idx(idx,N-1)
        else:
            idx=None
            rest_index=None

        x = self.proj(x)
        x = self.proj_drop(x)
        return x,idx,rest_index

class Block(nn.Module):

    def __init__(self, dim, num_heads, keep_rate=1.,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.keep_rate=keep_rate
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, keep_rate=keep_rate,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B,N,C=x.shape
        input_x=x
        x,index,rest_index=self.attn(self.norm1(x))
        x = input_x + self.drop_path(x)
        if self.keep_rate>0 and self.keep_rate<1:
            non_cls=x[:,1:]
            # left_tokens=torch.gather(non_cls, dim=1, index=index.unsqueeze(-1).expand(-1, -1, C))
            # fuse_tokens=torch.gather(non_cls, dim=1, index=fuse_index.unsqueeze(-1).expand(-1, -1, C))
            # fuse_tokens=torch.sum(fuse_tokens,dim=1)
            # fuse_tokens=fuse_tokens.unsqueeze(1)
            keep_part = torch.gather(non_cls, dim=1, index=index.unsqueeze(-1).expand(-1, -1, C))
            shuffle_part = torch.gather(non_cls, dim=1, index=rest_index.unsqueeze(-1).expand(-1, -1, C))

            random_part=shuffle_part
            b, n, d = random_part.shape
            random_part = random_part.reshape(b * n, d)
            random_part = random_part[torch.randperm(random_part.shape[0]), :]
            random_part = random_part.reshape(b, n, d)
            random_part = torch.cat((keep_part, random_part), dim=1)

            for bs in range(random_part.shape[0]):
                # random permutation
                random_part[bs] = random_part[bs][torch.randperm(random_part.shape[1]), :]

            x=torch.cat([x[:,0:1],random_part],dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
