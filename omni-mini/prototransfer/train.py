import os
import sys
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                           '../'))
import configs
from prototransfer.main import main
import argparse

# Parse command line / default arguments
parser = argparse.ArgumentParser(description='')

# Data parameters
parser.add_argument('--dataset', type=str, default='omniglot',  help='Dataset to use')
parser.add_argument('--datapath', type=str, default=configs.data_path,
                    help='Path, where datasets are stored')
parser.add_argument('--num_data_workers_cuda', type=int, default=8,
                    help='The number of workers for data loaders on GPU')
parser.add_argument('--num_data_workers_cpu', type=int, default=0,
                    help='The number of workers for data loaders on CPU')
parser.add_argument('--merge_train_val', action='store_true',
                    help='Merge training and validation set')


# Training parameters
parser.add_argument('--backbone', type=str, default='conv4',  help='Backbone architecture')
parser.add_argument('--distance', type=str, default='euclidean',
                    help='The distance metric to minimise.')
parser.add_argument('--train_support_shots', type=int, default=1,
                    help='Number of support images at training time')
parser.add_argument('--train_query_shots', type=int, default=3,
                    help='Number of query images at training time')
parser.add_argument('--epochs', type=int, default=10000,  help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--iterations', type=int, default=100, help='Number of \
                    iterations per epoch')
parser.add_argument('--lr_decay_step', type=int, default=25000,
                    help='Number of epochs after which to decay the learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help="Weight decay rate")
parser.add_argument('--patience', type=int, default=200,
                    help='Patience until early stopping. -1 means no early stopping')

# Self-supervised parameters
parser.add_argument('--self_sup_loss', type=str, default='proto',
                    help='The self-supervised loss to use.')
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size for ProtoTransfer')
parser.add_argument('--learn_temperature', action='store_true',
                    help='Whether to learn the softmax temperature.')
parser.add_argument('--n_images', type=int, default=None,
                    help='The number of images to use for training')
parser.add_argument('--n_classes', type=int, default=None,
                    help='The number of classes to use for training')
parser.add_argument('--no_aug_support', action='store_true',
                    help='The (single!) support example is not augmented')
parser.add_argument('--no_aug_query', action='store_true',
                    help='The (single!) query example is not augmented')

# Saving and loading parameters
parser.add_argument('--save', type=bool, default=True,  help='Whether to save the best model')
parser.add_argument('--save_path', type=str, default='',  help='Save path')
parser.add_argument('--load_path', type=str, default='', help='Optionally load a model')
parser.add_argument('--load_last', action='store_true',
                    help='Optionally load from default model path')
parser.add_argument('--load_best', action='store_true',
                    help='Optionally load from default best model path')

# Create arguments collection
args = parser.parse_args()

# Perform training and evaluation
main(args, mode='train')
