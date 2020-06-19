"""Adapted from https://github.com/tristandeleu/pytorch-meta/blob/master/examples/protonet/model.py"""

import torch.nn as nn
import torchvision.models as models

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class CNN_4Layer(nn.Module):
    def __init__(self, in_channels, out_channels=64, hidden_size=64):
        super(CNN_4Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[-3:]))
        return embeddings.view(*inputs.shape[:-3], -1)

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        blocks_and_layers = list(resnet18.children())[:-1]
        self.encoder = nn.Sequential(*blocks_and_layers)

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)

