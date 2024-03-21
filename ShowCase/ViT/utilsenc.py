import math

import torch
from einops import rearrange
#import model
import numpy as np
import privacy_account_gdp as gdp
from torchvision import transforms

def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis).to(tensor.device)  # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

def BatchPatchShuffle(x,k=0.1):
    row_perm = torch.rand((x.shape[0], x.shape[1])).argsort(1).to(x.device)
    percent = int(row_perm.shape[1] * k)
    for _ in range(x.ndim - 2): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(2)], *(x.shape[2:]))  # reformat this for the gather operation
    shuffle_part = x.gather(1, row_perm)
    keep_part = shuffle_part[:, :percent, :]

    random_part = shuffle_part[:, percent:, :]
    b, n, d = random_part.shape
    random_part = random_part.reshape(b * n, d)
    random_part = random_part[torch.randperm(random_part.shape[0]), :]
    random_part = random_part.reshape(b, n, d)
    input = torch.cat((keep_part, random_part), dim=1)
    perm_back = row_perm.argsort(1)
    x = input.gather(1, perm_back)

    # also shuffle patches of one batch
    #x=PatchShuffle(x)
    return x

def BatchPatchPartialShuffle(x,k1=0.1,k2=0.8):
    row_perm = torch.rand((x.shape[0], x.shape[1])).argsort(1).to(x.device)
    percent1 = int(row_perm.shape[1] * k1*k2)
    percent2 = int(row_perm.shape[1] * k1)
    for _ in range(x.ndim - 2): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(2)], *(x.shape[2:]))  # reformat this for the gather operation
    shuffle_part = x.gather(1, row_perm)
    keep_part = shuffle_part[:, :percent1, :]
    keep_shffule_part = shuffle_part[:, percent1:percent2, :]

    random_part = shuffle_part[:, percent2:, :]
    b, n, d = random_part.shape
    random_part = random_part.reshape(b * n, d)
    random_part = random_part[torch.randperm(random_part.shape[0]), :]
    random_part = random_part.reshape(b, n, d)

    random_part = torch.cat((keep_shffule_part,random_part),dim=1)
    random_part = PatchShuffle(random_part)

    input = torch.cat((keep_part, random_part), dim=1)
    perm_back = row_perm.argsort(1)
    x = input.gather(1, perm_back)
    return x


# k=1 99; k=0 97; k=0.5 98.9; k=0.25 98.2;k=0.15 97.8; k=0.3 98.4; k=0.35 98.6; k=0.4 98.7
def PatchPartialShuffle(x,k=0.1):
    row_perm = torch.rand((x.shape[0], x.shape[1])).argsort(1).to(x.device)
    percent = int(row_perm.shape[1] * k)
    for _ in range(x.ndim - 2): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(2)], *(x.shape[2:]))  # reformat this for the gather operation
    shuffle_part = x.gather(1, row_perm)
    keep_part = shuffle_part[:, :percent, :]

    random_part = shuffle_part[:, percent:, :]
    random_part=shufflerow(random_part,1)
    input = torch.cat((keep_part, random_part), dim=1)
    perm_back = row_perm.argsort(1)
    x = input.gather(1, perm_back)
    return x

def PatchShuffle(x):
    for bs in range(x.shape[0]):
        # random permutation
        x[bs] = x[bs][torch.randperm(x.shape[1]),:]
    return x

def PatchOcclusion(x):
    for bs in range(x.shape[0]):
        # each instance choose 0.51 percentage of tokens to 0
        x[bs][torch.randperm(x.shape[1])[int(x.shape[1]*0.51):],:] = 0 #cifar10 0.991
    return x

def PE_Loss(embedding):
    embedding=embedding[:,1:,:]
    a=int(math.sqrt(embedding.shape[1]))
    embedding=rearrange(embedding,'b (h w) d -> b h w d',h=a)
    h_sim=torch.cosine_similarity(embedding[:,:a-1,:,:],embedding[:,1:a:,:,:],dim=3)
    w_sim=torch.cosine_similarity(embedding[:,:,:a-1,:],embedding[:,:,1:a:,:],dim=3)
    h_sim_mean=torch.mean(h_sim)
    w_sim_mean=torch.mean(w_sim)
    sim=0.5*h_sim_mean+0.5*w_sim_mean
    return torch.abs(sim)

class blur():
    def __init__(self,size):
        super(blur, self).__init__()
        self.transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        print('The Blur size is: ',size)

    def process(self,ins):
        device=ins.device
        ins=ins.detach().cpu()
        pics=[]
        for i in range(ins.shape[0]):
            img=ins[i]
            img-=torch.min(img)
            img/=torch.max(img)
            pics.append(self.transform(img).unsqueeze(0))
        ins=torch.stack(pics,dim=0)[:,0,...]
        ins=ins.to(device)
        return ins

class add_gaussian_noise():
    def __init__(self,sigma):
        super(add_gaussian_noise, self).__init__()
        self.sigma=sigma
        print('The sigma is: ',self.sigma)

    def process(self,ins):
        noise = ins.data.new(ins.size()).normal_(0, self.sigma)
        return ins + noise


class add_gdp_noise():
    def __init__(self,epoch,sigma,N,batch_size,delta):
        super(add_gdp_noise, self).__init__()
        self.sigma=sigma
        eps=gdp.compute_eps_uniform(epoch,sigma,N,batch_size,delta)
        print('The Epsilon under this sigma is: ',eps)

    def process(self,ins):
        mean = torch.mean(ins)
        sigma = self.sigma
        clip_value = 2 * np.sqrt(ins.size()[0] * ins.size()[1] * ins.size()[2] * ins.size()[3])
        noise = ins.data.new(ins.size()).normal_(mean, sigma * clip_value)
        return ins + noise
        
class cut_mix():
    def __init__(self, prob):
        super(cut_mix, self).__init__()
        self.prob = prob
        print("CutMix Prob:", prob)
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

    # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
        
    def process(self, inputs, beta = 1.0):
        r = np.random.rand(1)
        if r<self.prob:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            #target_a = targets
            #target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            return inputs
        else: 
            return inputs

class identity():
    def __init__(self):
        super(identity, self).__init__()

    def process(self,ins):
        return ins



def change_GPUCPUstae():
    net = model.vit_pretrain(1).cuda()
    net.load_state_dict(torch.load('./checkpoints/vit_base_patch16_224_ImageNet2Cifar10_PatchShuffleInitPE&Loss.pth'))
    net=net.cpu()
    torch.save(net.state_dict(), 'PatchShufflePEInit&Loss.pth')

if __name__ == "__main__":
    GDP=add_gdp_noise(60,8,202599,50,1e-6)
