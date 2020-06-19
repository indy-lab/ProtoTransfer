import h5py
import os
import io
import numpy as np
from PIL import Image
import json

import torch
from torch.utils.data import Dataset#, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

import torch.nn.functional as F

class UnlabelledDataset(Dataset):
    def __init__(self, dataset, datapath, split, transform=None,
                 n_support=1, n_query=1, n_images=None, n_classes=None,
                 seed=10, no_aug_support=False, no_aug_query=False):
        """
        Args:
            dataset (string): Dataset name.
            datapath (string): Directory containing the datasets.
            split (string): The dataset split to load.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_support (int): Number of support examples
            n_query (int): Number of query examples
            no_aug_support (bool): Wheteher to not apply any augmentations to the support
            no_aug_query (bool): Wheteher to not apply any augmentations to the query
            n_images (int): Limit the number of images to load.
            n_classes (int): Limit the number of classes to load.
            seed (int): Random seed to for selecting images to load.
        """
        self.n_support = n_support
        self.n_query = n_query
        self.img_size = (28, 28) if dataset=='omniglot' else (84, 84)
        self.no_aug_support = no_aug_support
        self.no_aug_query = no_aug_query

        # Get the data or paths
        self.dataset = dataset
        self.data = self._extract_data_from_hdf5(dataset, datapath, split, 
                                                 n_classes, seed)

        # Optionally only load a subset of images
        if n_images is not None:
            random_idxs = np.random.RandomState(seed).permutation(len(self))[:n_images]
            self.data = self.data[random_idxs]

        # Get transform
        if transform is not None:
            self.transform = transform
        else:
            if self.dataset == 'cub':
                self.transform = transforms.Compose([
                    get_cub_default_transform(self.img_size),
                    get_custom_transform(self.img_size)])
                self.original_transform = transforms.Compose([
                    get_cub_default_transform(self.img_size),
                    transforms.ToTensor()])
            elif self.dataset == 'omniglot':
                self.transform = get_omniglot_transform((28, 28))
                self.original_transform = identity_transform((28, 28))
            else:
                self.transform = get_custom_transform(self.img_size)
                self.original_transform = identity_transform(self.img_size)

    def _extract_data_from_hdf5(self, dataset, datapath, split,
                                n_classes, seed):
        datapath = os.path.join(datapath, dataset)

        # Load omniglot
        if dataset == 'omniglot':
            classes = []
            with h5py.File(os.path.join(datapath, 'data.hdf5'), 'r') as f_data:
                with open(os.path.join(datapath,
                          'vinyals_{}_labels.json'.format(split))) as f_labels:
                    labels = json.load(f_labels)
                    for label in labels:
                        img_set, alphabet, character = label
                        classes.append(f_data[img_set][alphabet][character][()])
        # Load mini-imageNet
        else:
            with h5py.File(os.path.join(datapath, split + '_data.hdf5'), 'r') as f:
                datasets = f['datasets']
                classes = [datasets[k][()] for k in datasets.keys()]

        # Optionally filter out some classes
        if n_classes is not None:
            random_idxs = np.random.RandomState(seed).permutation(len(classes))[:n_classes]
            classes = [classes[i] for i in random_idxs]

        # Collect in single array
        data = np.concatenate(classes)
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.dataset == 'cub':
            image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        else:
            image = Image.fromarray(self.data[index])

        view_list = []
        
        
        for _ in range(self.n_support):
            if not self.no_aug_support:
                view_list.append(self.transform(image).unsqueeze(0))
            else:
                assert self.n_support == 1
                view_list.append(self.original_transform(image).unsqueeze(0))
        
        for _ in range(self.n_query):
            if not self.no_aug_query:
                view_list.append(self.transform(image).unsqueeze(0))
            else:
                assert self.n_query == 1
                view_list.append(self.original_transform(image).unsqueeze(0))
        
        return dict(data=torch.cat(view_list))

def get_cub_default_transform(size):
    return transforms.Compose([
        transforms.Resize([int(size[0] * 1.5), int(size[1] * 1.5)]),
        transforms.CenterCrop(size)])

def get_simCLR_transform(img_shape):
    """Adapted from https://github.com/sthalles/SimCLR/blob/master/data_aug/dataset_wrapper.py"""
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                          saturation=0.8, hue=0.2)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_shape[-2:]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                         # GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                          transforms.ToTensor()])
    return data_transforms

def get_omniglot_transform(img_shape):
    data_transforms = transforms.Compose([
                                          transforms.Resize(img_shape[-2:]),
                                          transforms.RandomResizedCrop(size=img_shape[-2:],
                                                                       scale=(0.6, 1.4)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda t: F.dropout(t, p=0.3)),
                                          transforms.RandomErasing()
                                          ])
    return data_transforms

def get_custom_transform(img_shape):
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                          saturation=0.4, hue=0.1)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_shape[-2:],
                                                                       scale=(0.5, 1.0)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor()])
    return data_transforms

def identity_transform(img_shape):
    return transforms.Compose([transforms.Resize(img_shape),
                               transforms.ToTensor()])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.pause(2)

    dataset = 'miniimagenet'
    dataset_path = '../semifew_data'
    split = 'train'
    dataset = UnlabelledDataset(dataset, dataset_path, split, size224=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    for batch in dataloader:
        img = make_grid(batch['data'], nrow=5)
        show(img)

