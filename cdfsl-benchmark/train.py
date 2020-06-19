import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.protoclr import ProtoCLR

from io_utils import model_dict, parse_args, get_resume_file  
from datasets import (miniImageNet_few_shot, ISIC_few_shot, CropDisease_few_shot,
                      EuroSAT_few_shot, Chest_few_shot)  


def train(base_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')     

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer) 

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'args':params}, outfile)
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimization = 'Adam'
    learn_temperature = False#True
    projection_head = None#'mlp'
    batch_size = params.batch_size#256
    method = params.method
    dataset = params.dataset
    print('Setting:', dataset, image_size, optimization, 'Temp:',
          learn_temperature, projection_head,"Method: ", method)

    # Get dataloader
    if dataset == 'miniImagenet':
        datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = batch_size)
    else:
        method = method + 'FromScratch'
    if dataset == 'ISIC':
        batch_size = 71
        datamgr = ISIC_few_shot.SimpleDataManager(image_size, batch_size = batch_size)
    elif dataset == 'CropDisease':
        datamgr = CropDisease_few_shot.SimpleDataManager(image_size, batch_size = batch_size)
    elif dataset == 'EuroSAT':
        datamgr = EuroSAT_few_shot.SimpleDataManager(image_size, batch_size = batch_size)
    elif dataset == 'Chest':
        datamgr = Chest_few_shot.SimpleDataManager(image_size, batch_size = batch_size)
 
    # Define data loader
    if params.custom_aug is not None:
        aug = params.custom_aug
        method += '_aug4' + aug
    else:
        aug = True
    base_loader = datamgr.get_data_loader(aug = aug,
                                          n_support=params.n_support, n_query=params.n_query,
                                          no_aug_support=params.no_aug_support, no_aug_query=params.no_aug_query)
    
    # Choose method
    if method.startswith('protoclr') or method.startswith('umtra'):
        model           = ProtoCLR(model_dict[params.model], shots=params.n_support)
    else:
        print("unknown method!")
    
    model = model.cuda()
    save_dir =  configs.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s%s_%s%s' %(save_dir, dataset,
                                                        params.model, method,
                                                        params.n_support, "s" if params.no_aug_support else "s_aug",
                                                        params.n_query, "q" if params.no_aug_query else "q_aug")
    params.checkpoint_dir += "_bs{}".format(params.batch_size)
    
    print('Saving in dir:', params.checkpoint_dir)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)
