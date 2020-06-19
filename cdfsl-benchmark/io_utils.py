import numpy as np
import os
import glob
import argparse
import backbone

model_dict = dict(
            ResNet10 = backbone.ResNet10
            )


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImagenet',        help='training base model')
    parser.add_argument('--model'       , default='ResNet10',      help='backbone architecture') 
    parser.add_argument('--method'      , default='protoclr',   help='protoclr/umtra') 
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_test_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') 
    parser.add_argument('--freeze_backbone'   , action='store_true', help='Freeze the backbone network for finetuning')
    parser.add_argument('--batch_size'      , default=50, type=int,  help='Batch size for non-meta-training')

    parser.add_argument('--models_to_use', '--names-list', nargs='+', default=['miniImageNet', 'caltech256', 'DTD', 'cifar100', 'CUB'], help='pretained model to use')
    parser.add_argument('--fine_tune_all_models'   , action='store_true',  help='fine-tune each model before selection') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--n_support' , default=1, type=int,help ='Number of support examples (1 for ProtoTransfer/UMTRA)')
    parser.add_argument('--n_query' , default=3, type=int,help ='Number of query examples (1 for UMTRA, >=1 for ProtoTransfer)')
    parser.add_argument('--no_aug_support', action='store_true', help='Whether to NOT augment the (single!) support example')
    parser.add_argument('--no_aug_query', action='store_true', help='Whether to NOT augment the (single!) query example') 
    

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=25, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') 
        parser.add_argument('--custom_aug', default=None, type=str, help ='Dataset specific augmentation') # for meta-learning methods, each epoch contains 100 episodes
    
    elif script == 'save_features':
        parser.add_argument('--split', default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--proto_init'  , action='store_true', help='initialize the adapted classifier via distances to prototypes')
        parser.add_argument('--lr_rate', default=0.001, type=float,help ='Learning rate for fine-tuning')
        parser.add_argument('--ft_steps', default=15, type=int,help ='Number of fine-tuning steps')
        
    else:
        raise ValueError('Unknown script')
        
    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
