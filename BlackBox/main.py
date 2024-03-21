import torch
from torch.autograd import Variable
import time
import copy
import numpy as np
import pytorch_lightning as pl

import DataLoaders
#import modeltrain
from models import t2t_vit_24
import trainer
#from utils import load_for_transfer_learning
#from utils import progress_bar
import utilsenc
import argparse

# arg parser
parser = argparse.ArgumentParser(description='Model Inversion Attack')

parser.add_argument('--transform', type=str, default='None',
                    help='gdp or blur or None')
parser.add_argument('--transform-value', type=float, default=0,
                    help='Parameter for gdp for blur transformation')
parser.add_argument('--R', action='store_true',
                    help='Row shuffle')
args = parser.parse_args()

def train_G(batch_size,max_epochs,net,wrapper):
    train_loader = DataLoaders.get_loader('celeba', '../../data', batch_size, 'train', num_workers=8, pin_memory=True)
    valid_loader = DataLoaders.get_loader('celeba','../../data', batch_size, 'valid', num_workers=8, pin_memory=True)

    net.total_steps = ((len(train_loader.dataset) // (batch_size)) // 1 * float(max_epochs))
    wrapper.fit(net, train_loader, valid_loader)
    #wrapper.test(net, valid_loader)

def train():
    batch_size=128
    max_epochs = 50
    train_loader = DataLoaders.get_loader('celeba', '../../data', batch_size, 'train', num_workers=8, pin_memory=True)
    
    transform=None
    if args.transform=='gdp':
        transform=utilsenc.add_gdp_noise(60,args.transform_value,len(train_loader), 128, 1.0)#1e-6)
    elif args.transform=='blur':
        transform = utilsenc.blur(args.transform_value)
    elif args.transform=='gaussian':
        transform= utilsenc.add_gaussian_noise(args.transform_value)
    elif args.transform=='low_pass':
        transform = utilsenc.low_pass_filter(args.transform_value)
    elif args.transform=='cutmix':
        transform = utilsenc.cutmix(args.transform_value)
    R = args.R

    checkpoints = './checkpoints/timm_pretrain_224_ImageNet2CelebA_'+args.transform+str(args.transform_value)+'_R_'+str(args.R)+'.pth'
    
    net = trainer.TrainWrapper(checkpoints=checkpoints,num_labels=10,k1=1.,k2=0,first_keep_rate=1.,transform=transform, transform_value = args.transform_value,transform_name=args.transform, R = R)
    wrapper = pl.Trainer(gpus=1,precision=32,max_epochs=max_epochs,default_root_dir='./log/',enable_progress_bar=0)
    train_G(batch_size,max_epochs,net,wrapper)

def valid_G_kernel(batch_size,net:trainer.TrainWrapper,wrapper:pl.Trainer):
    test_loader = DataLoaders.get_loader('celeba','../../data', batch_size, 'valid', num_workers=4, pin_memory=True,celeba_param='identity')
    wrapper.test(net,test_loader)

def valid_G():
    batch_size=8
    max_epochs = 50
    valid_loader = DataLoaders.get_loader('celeba','../../data', batch_size, 'valid', num_workers=8, pin_memory=True)
    
    transform=None
    if args.transform=='gdp':
        transform=utilsenc.add_gdp_noise(60,args.transform_value,len(train_loader), 128, 1.0)#1e-6)
    elif args.transform=='blur':
        transform = utilsenc.blur(args.transform_value)
    elif args.transform=='gaussian':
        transform= utilsenc.add_gaussian_noise(args.transform_value)
    elif args.transform=='low_pass':
        transform = utilsenc.low_pass_filter(args.transform_value)
    elif args.transform=='cutmix':
        transform = utilsenc.cutmix(args.transform_value)

    R = args.R

    checkpoints = './checkpoints/timm_pretrain_224_ImageNet2CelebA_'+args.transform+str(args.transform_value)+'_R_'+str(args.R)+'.pth'
    
    net = trainer.TrainWrapper(checkpoints='I should not exist',num_labels=10, k1=1.,k2=0, transform = transform, transform_value = args.transform_value, R = R, transform_name=args.transform)
    net.G.load_state_dict(torch.load(checkpoints))
    wrapper = pl.Trainer(gpus=1, precision=32, max_epochs=max_epochs, default_root_dir='./log/',enable_progress_bar=0)
    valid_G_kernel(batch_size, net, wrapper)

def valid():
    batch_size = 10
    test_loader = DataLoaders.get_loader('cifar10', "../../data/", batch_size, 'valid', num_workers=4,pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    F = t2t_vit_24()
    checkpoint = torch.load("../checkpoint_cifar10_T2t_vit_24/ckpt_0.05_0.0005_96.99_noPos.pth")
    F.load_state_dict(checkpoint['net'])
    F = modeltrain.t2t_pretrain(F, k1=1, k2=0)
    F.to(device)
    F.eval()
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = F(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #progress_bar(batch_idx, len(test_loader), 'Acc: %.3f%% (%d/%d)'% (100. * correct / total, correct, total))
 
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()
    valid_G()
