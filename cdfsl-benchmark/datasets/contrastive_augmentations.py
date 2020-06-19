from torchvision import transforms
from PIL import ImageFilter
import random
import torch.nn.functional as F

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Taken from https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    """

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_simCLR_transform(img_shape):
    """Adapted from https://github.com/sthalles/SimCLR/blob/master/data_aug/dataset_wrapper.py"""
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                          saturation=0.8, hue=0.2)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_shape[-2:]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur()])

def get_chestX_transform(img_shape):
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                          saturation=0, hue=0)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=img_shape[-2:]),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1),
                                shear=10),
        GaussianBlur(sigma=(0, 1.)),
        transforms.ToTensor(),
        #transforms.Lambda(lambda t: F.dropout(t, p=0.3)),
        #transforms.RandomErasing()
        ])
    return data_transforms
