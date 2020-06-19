# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import nn

class ProtoCLR(nn.Module):
    """Calculate the UMTRA-style loss on a batch of images.
    If shots=1 and only two views are served for each image,
    this corresponds exactly to UMTRA except that it uses ProtoNets
    instead of MAML.

    Parameters:
        - model_func: The encoder network.
        - shots: The number of support shots.
    """
    def __init__(self, model_func, shots=1):
        super(ProtoCLR, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.feature    = model_func()
        self.top1 = utils.AverageMeter()
        self.shots = shots

    def forward(self, x):
        # Treat the first dim as way, the second as shots
        ways = x.size(0)
        n_views = x.size(1)
        shots = self.shots
        query_shots = n_views - shots
        x_support = x[:,:shots].reshape((ways*shots, *x.shape[-3:]))
        x_support = Variable(x_support.cuda())
        x_query = x[:,shots:].reshape((ways*query_shots, *x.shape[-3:]))
        x_query = Variable(x_query.cuda())

        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(1) # shot dim
        y_query = y_query.repeat(1, query_shots)
        y_query = y_query.view(-1).cuda()

        # Extract features
        x_both = torch.cat([x_support, x_query], 0)
        z = self.feature(x_both)
        z_support = z[:ways*shots]
        z_query = z[ways*shots:]

        # Get prototypes
        z_proto = z_support.view(ways, shots, -1).mean(1) #the shape of z is [n_data, n_dim]

        # Calculate loss and accuracies
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores, y_query

    def forward_loss(self, x):
        scores, y = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y)
   
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
 

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
