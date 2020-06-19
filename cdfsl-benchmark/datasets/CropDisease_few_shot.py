# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *
from datasets.miniImageNet_few_shot import TransformLoader as MiniImTransformLoader
from torchvision.datasets.folder import default_loader

identity = lambda x:x

def identity_transform(img_shape):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    return data_transforms

class SimpleDataset:
    def __init__(self, transform, original_transform,
                target_transform=identity,
                n_support=1, n_query=1,
                no_aug_support=False, no_aug_query=False,
                img_size=(224,224)):
        self.transform = transform
        self.target_transform = target_transform
        
        self.img_size = img_size
        self.original_transform = original_transform
        
        self.n_support = n_support
        self.n_query = n_query
        
        self.no_aug_support = no_aug_support
        self.no_aug_query = no_aug_query

        # Adaptation to unlabelled dataset
        image_path = CropDisease_path + "/dataset/train/"
        with open('unsupervised-track/UNSUPERVISED_CROPDISEASE.txt') as f:
            image_names = f.readlines()
        self.image_paths = [image_path + n.strip() for n in image_names]
        
    def __getitem__(self, i):
        # Loading from paths instead of RAM
        data = default_loader(self.image_paths[i])
        
        view_list = []
        
        
        for _ in range(self.n_support):
            if not self.no_aug_support:
                view_list.append(self.transform(data).unsqueeze(0))
            else:
                assert self.n_support == 1
                view_list.append(self.original_transform(data).unsqueeze(0))
        
        for _ in range(self.n_query):
            if not self.no_aug_query:
                view_list.append(self.transform(data).unsqueeze(0))
            else:
                assert self.n_query == 1
                view_list.append(self.original_transform(data).unsqueeze(0))
        
        return torch.cat(view_list)

    def __len__(self):
        return len(self.image_paths)

class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(38)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        # Dataset over image paths instead of images
        d = ImageFolder(CropDisease_path + "/dataset/train/", loader=lambda path: path)

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)
    

        #for key, item in self.sub_meta.items():
        #    print (len(self.sub_meta[key]))

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        # Load image from path, instead of preloading (saves RAM)
        img = self.transform(default_loader(self.sub_meta[i]))
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        #self.trans_loader = TransformLoader(image_size)
        self.trans_loader = MiniImTransformLoader(image_size)

    def get_data_loader(self, aug, n_support=1, n_query=1, no_aug_support=False, no_aug_query=False): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        original_transform = self.trans_loader.get_composed_transform(aug=False)
        dataset = SimpleDataset(transform=transform,
                                original_transform=original_transform,
                                n_support=n_support, n_query=n_query,
                                no_aug_support=no_aug_support, no_aug_query=no_aug_query)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':

    train_few_shot_params   = dict(n_way = 5, n_support = 5) 
    base_datamgr            = SetDataManager(224, n_query = 16)
    base_loader             = base_datamgr.get_data_loader(aug = True)

    cnt = 1
    for i, (x, label) in enumerate(base_loader):
        if i < cnt:
            print(label)
        else:
            break
