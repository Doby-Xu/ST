import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import os
import argparse

import utilsenc

import math
import DataLoaders
from timm.models import *
#from utils import progress_bar
from timm.models import create_model
#from utils import load_for_transfer_learning
from ViT_base_square.vit_timm import VisionTransformer as vit
import timm_pretrain

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='cifar10 or cifar100 or celeba or imdb')
parser.add_argument('--b', type=int, default=128,
                    help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
# Transfer learning
parser.add_argument('--transfer-learning', default=False,
                    help='Enable transfer learning')
parser.add_argument('--transfer-model', type=str, default="./checkpoints/T2t_vit_24_pretrained.pth.tar",
                    help='Path to pretrained model for transfer learning')
parser.add_argument('--transfer-ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--epoch', type=int, default=60, metavar='N',
                    help='Training Epoch')

# obfuscation transform
parser.add_argument('--transform', type=str, default='None',
                    help='gdp or blur or None')
parser.add_argument('--transform-value', type=float, default=0,
                    help='Parameter for gdp for blur transformation')

parser.add_argument('--model', default='timm_pretrain', type=str, metavar='MODEL',
                    help='Name of model to train (default: "timm_pretrain"')
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
print("\n\n")
print(device)
print("\n\n")

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
else:
    print('Please use cifar10 or cifar100 or celeba dataset.')

data_root=args.data

trainloader=DataLoaders.get_loader(args.dataset,data_root,args.b,attr='train',num_workers=8)
testloader=DataLoaders.get_loader(args.dataset,data_root,args.b,attr='valid',num_workers=8)

print(f'learning rate:{args.lr}, weight decay: {args.wd}')

print('==> Building model..')


if args.model == "timm_pretrain":
    net=timm_pretrain.timm_pretrain(RS=args.R+args.RC, CS=0, num_classes=args.num_classes)




net = net.to(device)

# checkpoint = torch.load("./checkpoint_celeba_T2t_vit_24_1.0_1.0/ckpt_0.1_0.0005_88.2114827305884.pth")
# net.load_state_dict(checkpoint['net'])

if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='imdb' or args.dataset=='svhn':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()
    
# transform
transform=None
if args.transform=='gdp':
    transform=utilsenc.add_gdp_noise(args.epoch,args.transform_value,len(trainloader),args.b,1.0)#1e-6)
elif args.transform=='blur':
    transform = utilsenc.blur(args.transform_value)
elif args.transform=='gaussian':
    transform= utilsenc.add_gaussian_noise(args.transform_value)
elif args.transform=='low_pass':
    transform = utilsenc.low_pass_filter(args.transform_value)
elif args.transform=='cutmix':
    transform = utilsenc.cutmix(args.transform_value)

# set optimizer
if args.model == "timm_pretrain":
    parameters = [{'params': net.model.patch_embed.parameters()},
                {'params': net.pos_embed},
                {'params': net.model.blocks.parameters(), 'lr': args.transfer_ratio * args.lr},
                {'params': net.model.head.parameters()}]


    optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.epoch)
elif args.model == "vit_base_square":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.epoch)
elif args.model == "svit_base":
    parameters = [{'params': net.MLP.parameters()},
                {'params': net.pos_embed},
                {'params': net.model.blocks.parameters(), 'lr': args.transfer_ratio * args.lr},
                {'params': net.model.head.parameters()}]


    optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.epoch)

log_loss=[]
log_acc=[]

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.dataset=='celeba':
            targets=targets.float()
        if args.transform == "cutmix":
            inputs, target_a, target_b, lam = transform.process(inputs, targets)
            
            outputs = net(inputs)
            
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

        else:
            if transform is not None:
                inputs = transform.process(inputs)
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs= net(inputs)
                
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if args.dataset=='cifar10' or args.dataset=='cifar100' or args.dataset=='imdb' or args.dataset=='svhn':
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        else:
            predicted = (outputs > 0.5).long()
            correct += predicted.eq(targets).float().mean(dim=1).sum().item()
        total += targets.size(0)

        # You can't use it when running on background or Windows
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    global log_loss
    log_loss.append(train_loss/(batch_idx+1))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if transform != None and args.transform != "cutmix":
                inputs = transform.process(inputs)
                
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

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'checkpoint_{args.dataset}_{args.model}'):
            os.mkdir(f'checkpoint_{args.dataset}_{args.model}')
        shuffle_flag = "None"
        if args.R:
            shuffle_flag = "R"
        if args.RC:
            shuffle_flag = "RC"
        
        torch.save(state, f'./checkpoint_{args.dataset}_{args.model}/ckpt_{args.lr}_{shuffle_flag}_{args.transform}_{args.transform_value}.pth')
        best_acc = acc
        #torch.save(net.pos_embed,'pos_embed.pth')

    global log_acc
    log_acc.append(acc)

def confusion(prediction, truth):
    confusion_vector = prediction / truth
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    return true_positives, false_positives, true_negatives, false_negatives


for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
# np.savetxt('acc.txt',np.array(log_acc))
# np.savetxt('loss.txt',np.array(log_loss))
# valid_mcc()

