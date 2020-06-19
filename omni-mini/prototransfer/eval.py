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
                    help='Path, where dataset are stored')
parser.add_argument('--mode', type=str, default='test', help='Test on "test" or "val"')
parser.add_argument('--num_data_workers_cuda', type=int, default=8,
                    help='The number of workers for data loaders on GPU')
parser.add_argument('--num_data_workers_cpu', type=int, default=0,
                    help='The number of workers for data loaders on CPU')
parser.add_argument('--merge_train_val', action='store_true',
                    help='Whether to train on both the training and validation data')

# Test few-shot parameters
parser.add_argument('--train_ways', type=int, default=5,  help='Training ways')
parser.add_argument('--eval_ways', type=int, default=5,  help='Training ways')
parser.add_argument('--eval_support_shots', type=int, default=5,  help='Number of support images at test time')
parser.add_argument('--eval_query_shots', type=int, default=15,  help='Number of query images at test time')
parser.add_argument('--test_iterations', type=int, default=600, help='Number of \
                    iterations for the test epoch')

# Training parameters
parser.add_argument('--backbone', type=str, default='conv4',  help='Backbone architecture')
parser.add_argument('--distance', type=str, default='euclidean',
                    help='The distance metric to minimise.')

# Saving and loading parameters
parser.add_argument('--self_sup_loss', type=str, default='proto',
                    help='The self-supervised loss to use.')
parser.add_argument('--save', type=bool, default=True,  help='Whether to save the best model')
parser.add_argument('--load_path', type=str, default='', help='Optionally load a model')

# Supervised finetuning parameters
parser.add_argument('--sup_finetune', action='store_true',
                    help='Whether to finetune on the test data using a supervised loss.')
parser.add_argument('--sup_finetune_lr', type=float, default=0.001,
                    help='Learning rate for finetuning.')
parser.add_argument('--sup_finetune_epochs', type=int, default=15,
                    help='How many epochs to finetune.')
parser.add_argument('--ft_freeze_backbone', action='store_true',
                    help='Whether to freeze the backbone during finetuning.')
parser.add_argument('--finetune_batch_norm', action='store_true',
                    help='Whether to update the batch norm parameters during finetuning.')


# Create arguments collection
args = parser.parse_args()

# Adjust train args if evaluating with a ProtoCLR loss on training set
if args.mode == 'trainval':
    args.n_images = None
    args.n_classes = None
    args.train_support_shots = 1
    args.train_query_shots = 1
    args.no_aug_support = True
    args.no_aug_query = False
    args.batch_size = 100

# Perform training and evaluation
main(args, mode=args.mode)
