import numpy as np
from PIL import Image
import os
import h5py
from models import CNN_4Layer
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from torch.autograd import Variable

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                           '../'))
import configs

class LabelledDataset(Dataset):
    def __init__(self, dataset, datapath, split, class_keys):
        """
        Args:
            dataset (string): Dataset name.
            datapath (string): Directory containing the datasets.
            split (string): The dataset split to load.
            class_keys (list of string): Class labels to load.
        """
        self.img_size = (28, 28) if dataset=='omniglot' else (84, 84)

        # Get the data or paths
        self.dataset = dataset
        self.data, self.labels = self._extract_data_from_hdf5(dataset, datapath,
                                                              split, class_keys)

        if self.dataset == 'cub':
            self.transform = transforms.Compose([
                get_cub_default_transform(self.img_size),
                transforms.ToTensor()])
        else:
            self.transform = identity_transform(self.img_size)

    def _extract_data_from_hdf5(self, dataset, datapath, split,
                                class_keys):
        datapath = os.path.join(datapath, dataset)

        # Load mini-imageNet or CUB
        with h5py.File(os.path.join(datapath, split + '_data.hdf5'), 'r') as f:
            datasets = f['datasets']
            classes = [datasets[k][()] for k in class_keys]
            labels = [np.repeat([i], len(datasets[k][()])) for i, k in enumerate(class_keys)]

        # Collect in single array
        data = np.concatenate(classes)
        labels = np.concatenate(labels)
        return data, labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.dataset == 'cub':
            image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        else:
            image = Image.fromarray(self.data[index])

        image = self.transform(image)
        label = torch.tensor(self.labels[index])
        return (image, label)

def get_cub_default_transform(size):
    return transforms.Compose([
        transforms.Resize([int(size[0] * 1.5), int(size[1] * 1.5)]),
        transforms.CenterCrop(size)])

def identity_transform(img_shape):
    return transforms.Compose([transforms.Resize(img_shape),
                               transforms.ToTensor()])

def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)

        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    train_classes = ["n02687172", "n04251144", "n02823428", "n03676483", "n03400231"]
    test_classes = ["n03272010", "n07613480", "n03775546", "n03127925", "n04146614"]
    trainset = LabelledDataset('miniimagenet', configs.data_path,
                               'train', train_classes)
    testset = LabelledDataset('miniimagenet', configs.data_path,
                              'test', test_classes)
    trainloader = DataLoader(trainset, shuffle=False, batch_size=100)
    testloader = DataLoader(testset, shuffle=False, batch_size=100)

    # Load checkpoint
    model = CNN_4Layer(in_channels=3)
    load_path = 'prototransfer/checkpoints/protoclr/proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar'
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    print("Loaded checkpoint '{}' (epoch {})"
          .format(load_path, start_epoch))

    model.cuda()
    model.eval()
    print('----------------- Save train features ------------------------')
    save_features(model, trainloader, 'plots/featuresProtoCLR_mini-ImageNet_train.hdf5')
    print('----------------- Save test features ------------------------')
    save_features(model, testloader, 'plots/featuresProtoCLR_mini-ImageNet_test.hdf5')

