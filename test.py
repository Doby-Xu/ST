import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import os
import argparse

import math
import DataLoaders
from timm.models import *
from utils import progress_bar

from ViT_base_square.vit_timm import VisionTransformer as vit
import timm_pretrain

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='cifar10 or cifar100 or celeba or imdb')
parser.add_argument('--b', type=int, default=64,
                    help='batch size')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')

parser.add_argument('--model', default='vit_base_square', type=str, metavar='MODEL',
                    help='Name of model to train (default: "vit_base_square"')
parser.add_argument('--R', action='store_true',
                    help='Row shuffle')
parser.add_argument('--RC', action='store_true',
                    help='Row and Column shuffle')

parser.add_argument('--data', default='../data', type=str, 
                    help='data path')

parser.add_argument('--ckpt', default='./checkpoints/ckpt.pth', type=str, 
                    help='checkpoint path')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

if args.dataset=='cifar10':
    args.num_classes = 10
elif args.dataset=='svhn':
    args.num_classes = 10
elif args.dataset=='cifar100':
    args.num_classes = 100
elif args.dataset=='celeba':
    args.num_classes = 40
elif args.dataset=='imdb':
    args.num_classes = 2
else:
    print('Please use cifar10 or cifar100 or celeba dataset.')

data_root=args.data
from ViT_base_square.vit_timm import VisionTransformer as vit
#trainloader=DataLoaders.get_loader(args.dataset,data_root,args.b,attr='train',num_workers=8)
testloader=DataLoaders.get_loader(args.dataset,data_root,args.b,attr='valid',num_workers=8)

print('==> Building model..')

if args.model == "timm_pretrain":
    #This model is not square, RCS can not be used
    net=timm_pretrain.timm_pretrain(RS=args.R+args.RC, CS=0, num_classes=args.num_classes)
    #checkpoint = torch.load(args.ckpt)
    #net.load_state_dict(checkpoint['net'])
elif args.model == "vit_base_square":
    print("using vit_base_square")
    #This model is basically the vit in timm. Let num_heads=1 and mlp_ratio = 1 to make it square
    net = vit(num_heads=1, mlp_ratio=1., num_classes = 40, RS=args.R+args.RC, CS=args.RC)
    # raise RuntimeError("Pre-trained model coming soon. Please do not use this model for now")
    checkpoint = torch.load(args.ckpt)
    net.load_state_dict(checkpoint['net'])



net = net.to(device)



if device == 'cuda':
    cudnn.benchmark = True

    
if args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='imdb' or args.dataset=='svhn':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

log_loss=[]
log_acc=[]
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs, targets
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if args.dataset == 'celeba':
                targets = targets.float()
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            if args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='imdb' or args.dataset=='svhn':
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            else:
                predicted = (outputs > 0.5).long()
                correct += predicted.eq(targets).float().mean(dim=1).sum().item()
            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



test(0)
