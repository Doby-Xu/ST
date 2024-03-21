import os
import torch
from torchvision import datasets, transforms
import torchaudio

def MnistTransform():
    transform=transforms.Compose([
                   transforms.ToTensor(),
                   # Normalize输入为两个tuple，output=(input-mean)/std
                   transforms.Normalize((0.13066,), (0.30811,)) # (x,)输出为一维tuple
               ])
    return transform

def CIFARTransform(attr):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if attr=='train':
        transform=train_transform
        print('train')
    else:
        transform=valid_transform
        print('valid')

    return transform

def CelebTransform(attr):
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    valid_transform = transforms.Compose([
        #transforms.Resize((16,16)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if attr=='train':
        transform=train_transform
        print('train')
    else:
        transform=valid_transform
        print('valid')

    return transform

def tinyTransform(attr):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                                 ])

    valid_transform =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                                 ])

    if attr=='train':
        transform=train_transform
        print('train')
    else:
        transform=valid_transform
        print('valid')

    return transform

def get_loader_speechcommand(shuffle,train,num_workers,pin_memory,batch_size,path):
    if train==True:
        subset='training'
    else:
        subset='validation'
    dataset=torchaudio.datasets.SPEECHCOMMANDS(root=path,download=True,subset=subset)
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

def get_loader_tiny(shuffle,train,num_workers,pin_memory,batch_size,path=
                    "/home/stu55/Tiny-ImageNet-Classifier/images/224/"):
    if train==True and shuffle==True:
        transform=tinyTransform('train')
    else:
        transform=tinyTransform('valid')
    
    if train==True:
        path=path+'train/'
    else:
        path=path+'val/'
    dataset = datasets.ImageFolder(path, transform=transform)
    ds=[]
    if train==True:
        interval=500
        choose=500
    else:
        interval=50
        choose=50
    for i in range(0,20):
        for j in range(0,choose):
            im,l=dataset[i*interval+j]
            ds.append((im,l))
    dataset=ds
    loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return loader
    
def get_loader(name,path,batch_size,attr='train',num_workers=0,pin_memory=False, celeba_param='attribute',**kwargs):

    if attr=='train':
        shuffle=True
        train=True
    elif attr=='valid':
        shuffle=False
        train=False
    else:
        shuffle=True
        train=False

    if name=='celeba':
        name=name+'_'+celeba_param

    if name=='mnist':
        dataloader=torch.utils.data.DataLoader(
                datasets.MNIST(path, train=train, download=True,
                               transform=MnistTransform()),
                               batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, pin_memory=pin_memory, drop_last=True
                               )
    elif name=='cifar10':
        dataloader=torch.utils.data.DataLoader(
                datasets.CIFAR10(path, train=train, download=True,
                               transform=CIFARTransform(attr)),
                               batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, pin_memory=pin_memory, drop_last=True
                               )
    elif name=='celeba_identity':
        dataloader=torch.utils.data.DataLoader(
                datasets.CelebA(path, split=attr, target_type='identity',download=True,
                               transform=CelebTransform(attr)),
                               batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, pin_memory=pin_memory, drop_last=True
                               )
    elif name=='celeba_attribute':
        dataloader=torch.utils.data.DataLoader(
                datasets.CelebA(path, split=attr, target_type='attr',download=True,
                               transform=CelebTransform(attr)),
                               batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, pin_memory=pin_memory, drop_last=True
                               )
    elif name=='lfwa':
        if attr=='valid':
            attr='test'
        dataloader = torch.utils.data.DataLoader(
            datasets.LFWPeople(path, split=attr,download=True,
                            transform=CelebTransform(attr)),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
    elif name=='svhn':
        if attr=='valid':
            attr='test'
        dataloader=torch.utils.data.DataLoader(
                datasets.SVHN(path, split=attr, download=True,
                               transform=CIFARTransform(attr)),
                               batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, pin_memory=pin_memory
                               )
    elif name=='speechcommand':
        dataloader=get_loader_speechcommand(shuffle,train,num_workers,pin_memory,batch_size,path)
    elif name=='tiny_imagenet':
        if os.path.exists(path+'/tiny/'):
            if attr=='train':
                dataloader=torch.load(path+'/tiny/tiny_train.pt')
            else:
                dataloader=torch.load(path+'/tiny/tiny_test.pt')
        else:
            dataloader=get_loader_tiny(shuffle,train,num_workers,pin_memory,batch_size,path)
    elif name=='imdb':
        if os.path.exists(path+'/IMDB/'):
            if attr=='train':
                dataloader=torch.load(path+'/IMDB/imdb_train.pt')
            else:
                dataloader=torch.load(path+'/IMDB/imdb_test.pt')
        else:
            from keras.preprocessing.sequence import pad_sequences
            from keras.datasets import imdb
            (x_train,y_train),(x_test, y_test)=imdb.load_data(num_words=10000)
            x_train = pad_sequences(x_train,maxlen=200,padding="post",truncating="post")
            x_test = pad_sequences(x_test,maxlen=200, padding="post", truncating="post")

            train_data = torch.utils.data.TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
            test_data = torch.utils.data.TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

            train_sampler = torch.utils.data.RandomSampler(train_data)
            train_loader = torch.utils.data.DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

            test_sampler = torch.utils.data.SequentialSampler(test_data)
            test_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
            if attr=='train':
                dataloader=train_loader
            else:
                dataloader=test_loader
        
    return dataloader

class loader_iter_lfwa(object):
    def __init__(self,batch_size=50):
        super(loader_iter_lfwa, self).__init__()
        self.loader=get_loader('lfwa', '../../data', batch_size, 'train', num_workers=4, pin_memory=True)
        self.loader_wrapper=iter(self.loader)
        self.back,_=next(self.loader_wrapper)

    def next_batch(self,bs):
        while self.back.shape[0]<bs:
            try:
                tmp,_=next(self.loader_wrapper)
            except:
                self.loader_wrapper=iter(self.loader)
                tmp, _ = next(self.loader_wrapper)
            self.back=torch.concat((self.back,tmp),dim=0)
        return_thing=self.back[:bs,...]
        self.back=self.back[bs:,...]
        return return_thing

if __name__=='__main__':
    train_loader=get_loader('speechcommand','../../../data',1,'train',4,True)
    for waveform, sample_rate, label, speaker_id, utterance_number in train_loader:
        print(waveform.shape,label)


