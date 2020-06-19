import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

from tqdm import tqdm
import sys
sys.path.append('..')
from methods.protonet import euclidean_dist

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _set_params(self, weight, bias):
        state_dict = dict(weight=weight, bias=bias)
        self.fc.load_state_dict(state_dict)
        #self.fc.weight.data = weight
        #self.fc.bias.data = bias

    def init_params_from_prototypes(self, z_support, n_way, n_support):
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(n_way, n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        # Interpretation of ProtoNet as linear layer (see Snell et al. (2017))
        self._set_params(weight=2*z_proto, bias=-torch.norm(z_proto, dim=-1)**2)


class ProtoClassifier():
    def __init__(self, n_way, n_support, n_query):
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

    def __call__(self, z_support, y_support, z_query):
        # Copied from methods/protonet.py "ProtoNet.set_forward()"
        # y_support is ignored (only for compatibility)
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


def finetune(novel_loader, n_query = 15, freeze_backbone = False,
             n_way = 5, n_support = 5, loadpath = '', adaptation = False,
             pretrained_dataset = 'miniImagenet', proto_init = False):
    correct = 0
    count = 0

    iter_num = len(novel_loader) 

    acc_all = []
    
    with tqdm(enumerate(novel_loader), total=len(novel_loader)) as pbar:

        for _, (x, y) in pbar:#, position=1,
                              #leave=False):

            ###############################################################################################
            # load pretrained model on miniImageNet
            pretrained_model = model_dict[params.model]()
            checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s%s_%s%s' %(configs.save_dir, params.dataset,
                                                                    params.model, params.method,
                                                                    params.n_support, "s" if params.no_aug_support else "s_aug",
                                                                    params.n_query, "q" if params.no_aug_query else "q_aug")
            checkpoint_dir += "_bs{}".format(params.batch_size)

            if params.save_iter != -1:
                modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
            elif params.method in ['baseline', 'baseline++'] :
                modelfile   = get_resume_file(checkpoint_dir)
            else:
                modelfile   = get_best_file(checkpoint_dir)

            tmp = torch.load(modelfile)
            state = tmp['state']

            state_keys = list(state.keys())
            for _, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)

            pretrained_model.load_state_dict(state)
            pretrained_model.cuda()
            pretrained_model.train()
            ###############################################################################################

            if adaptation:
                classifier = Classifier(pretrained_model.final_feat_dim, n_way)
                classifier.cuda()
                classifier.train()
            else:
                classifier = ProtoClassifier(n_way, n_support, n_query)

            ###############################################################################################
            n_query = x.size(1) - n_support
            x = x.cuda()
            x_var = Variable(x)

            batch_size = n_way
            support_size = n_way * n_support 

            y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

            x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) 
            x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
            pretrained_model.eval()
            z_a_i = pretrained_model(x_a_i.cuda())
            pretrained_model.train()

             ###############################################################################################
            loss_fn = nn.CrossEntropyLoss().cuda()
            if adaptation:
                inner_lr = params.lr_rate
                if proto_init: # Initialise as distance classifer (distance to prototypes)
                    classifier.init_params_from_prototypes(z_a_i,
                                                           n_way, n_support)
                #classifier_opt = torch.optim.SGD(classifier.parameters(), lr = inner_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
                classifier_opt = torch.optim.Adam(classifier.parameters(), lr = inner_lr)


                if freeze_backbone is False:
                    delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = inner_lr)

                total_epoch = params.ft_steps

                if freeze_backbone is False:
                    pretrained_model.train()
                else:
                    pretrained_model.eval()

                classifier.train()

                #for epoch in range(total_epoch):
                for epoch in tqdm(range(total_epoch), total=total_epoch, leave=False):
                    rand_id = np.random.permutation(support_size)

                    for j in range(0, support_size, batch_size):
                        classifier_opt.zero_grad()
                        if freeze_backbone is False:
                            delta_opt.zero_grad()

                        #####################################
                        selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()

                        z_batch = x_a_i[selected_id]
                        y_batch = y_a_i[selected_id] 
                        #####################################

                        output = pretrained_model(z_batch)
                        output = classifier(output)
                        loss = loss_fn(output, y_batch)

                        #####################################
                        loss.backward()

                        classifier_opt.step()

                        if freeze_backbone is False:
                            delta_opt.step()

                classifier.eval()

            pretrained_model.eval()

            output = pretrained_model(x_b_i.cuda())
            if adaptation:
                scores = classifier(output)
            else:
                scores = classifier(z_a_i, y_a_i, output)

            y_query = np.repeat(range( n_way ), n_query )
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()

            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            #print (correct_this/ count_this *100)
            acc_all.append((correct_this/ count_this *100))

            ###############################################################################################
            
            pbar.set_postfix(avg_acc=np.mean(np.asarray(acc_all)))

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('test')

    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_test_shot) 

    freeze_backbone = params.freeze_backbone
    ##################################################################

    dataset_names = ["ISIC", "EuroSAT", "CropDisease", "ChestX"]
    novel_loaders = []

    loader_name         = "ISIC"
    print ("Loading {}".format(loader_name))
    datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    
    #novel_loaders.append((loader_name, novel_loader))
    
    loader_name         = "EuroSAT"
    print ("Loading {}".format(loader_name))
    datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    
    #novel_loaders.append((loader_name, novel_loader))

    
    loader_name         = "CropDisease"
    print ("Loading {}".format(loader_name))
    datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    
    #novel_loaders.append((loader_name, novel_loader))
    
    loader_name         = "ChestX"
    print ("Loading {}".format(loader_name))
    datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append((loader_name, novel_loader))
    
    #########################################################################
    # Print checkpoint path to be loaded
    checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s%s_%s%s' %(configs.save_dir, params.dataset,
                                                                params.model, params.method,
                                                                params.n_support, "s" if params.no_aug_support else "s_aug",
                                                                params.n_query, "q" if params.no_aug_query else "q_aug")
    checkpoint_dir += "_bs{}".format(params.batch_size)

    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
    elif params.method in ['baseline', 'baseline++'] :
        modelfile   = get_resume_file(checkpoint_dir)
    else:
        modelfile   = get_best_file(checkpoint_dir)
    print('Evaluation from checkpoint:', modelfile)

    # Perform evaluation
    for idx, (loader_name, novel_loader) in enumerate(novel_loaders): 
    #for idx, novel_loader in tqdm(enumerate(novel_loaders), total=len(novel_loaders), position=0):
        print ('Dataset: ', loader_name)
        print ('Pretraining Dataset: ', params.dataset)
        print('Adaptation? ', params.adaptation)
        if params.adaptation:
            print (' --> Freeze backbone?', freeze_backbone)
            print (' --> Init classifier via prototypes?', params.proto_init)
            print (' --> Adaptation steps: ', params.ft_steps)
            print (' --> Adaptation learning rate: ', params.lr_rate)
        
        # replace finetine() with your own method
        finetune(novel_loader, n_query = 15, adaptation=params.adaptation,
                 freeze_backbone=freeze_backbone, proto_init=params.proto_init,
                 pretrained_dataset=params.dataset, **few_shot_params)
