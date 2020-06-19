import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR

import numpy as np
import copy
import scipy.stats as st
from tqdm import tqdm
import shutil
import os

from prototransfer.protonet import Protonet
from prototransfer.models import CNN_4Layer
from prototransfer.protoclr import ProtoCLR
from prototransfer.unlabelled_loader import UnlabelledDataset
from prototransfer.supervised_finetuning import supervised_finetuning
from prototransfer.episodic_loader import get_episode_loader

def main(args, mode='train'):
    ##########################################
    # Setup
    ##########################################

    # Check whether GPU is available
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Used device:', device)

    # Define datasets and loaders
    num_workers = args.num_data_workers_cuda if cuda else args.num_data_workers_cpu
    kwargs = {'num_workers': num_workers}

    # Load data for training
    if (mode == 'train') or (mode == 'trainval'):
        dataset_train = UnlabelledDataset(args.dataset,
                                          args.datapath, split='train',
                                          transform=None,
                                          n_images=args.n_images,
                                          n_classes=args.n_classes,
                                          n_support=args.train_support_shots,
                                          n_query=args.train_query_shots,
                                          no_aug_support=args.no_aug_support,
                                          no_aug_query=args.no_aug_query)

        # Optionally add validation set to training
        if args.merge_train_val:
            dataset_val = UnlabelledDataset(args.dataset, args.datapath, 'val',
                                            transform=None,
                                            n_support=args.train_support_shots,
                                            n_query=args.train_query_shots,
                                            no_aug_support=args.no_aug_support,
                                            no_aug_query=args.no_aug_query)

            dataset_train = ConcatDataset([dataset_train, dataset_val])

        # Train data loader
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=torch.cuda.is_available())

        if mode == 'trainval': # USe training set for ProtoCLR evaluation
            dataloader_test = dataloader_train

    # Test on the validation set
    elif mode == 'val': 
        dataloader_test = get_episode_loader(args.dataset, args.datapath,
                                             ways=args.eval_ways,
                                             shots=args.eval_support_shots,
                                             test_shots=args.eval_query_shots,
                                             batch_size=1,
                                             split='val',
                                             **kwargs)
        # The rest is identical to mode == test
        mode = 'test'

    # Load data for testing
    elif mode == 'test': 
        dataloader_test = get_episode_loader(args.dataset, args.datapath,
                                             ways=args.eval_ways,
                                             shots=args.eval_support_shots,
                                             test_shots=args.eval_query_shots,
                                             batch_size=1,
                                             split='test',
                                             **kwargs)

    # Define model
    ## Encoder
    if args.dataset in ['omniglot', 'doublemnist', 'triplemnist']:
        channels = 1
    elif args.dataset in ['miniimagenet', 'tieredimagenet', 'cub', 'cifar_fs']:
        channels = 3
    else:
        raise ValueError('No such dataset')
    if (args.backbone == 'conv4') or (args.backbone == 'cnn'):
        encoder = CNN_4Layer(in_channels=channels)
    else:
        raise ValueError('No such model')
    encoder = encoder.to(device)

    ## Protonet + Loss
    proto = Protonet(encoder, distance=args.distance, device=device)
    if (mode == 'train') or (mode == 'trainval'):
        self_sup_loss = ProtoCLR(encoder, n_support=args.train_support_shots,
                                 n_query=args.train_query_shots,
                                 device=device, distance=args.distance)

    if mode == 'train':
        # Define optimisation parameters
        optimizer = Adam(self_sup_loss.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

        # Save path
        os.makedirs('prototransfer/checkpoints', exist_ok=True)
        default_path = os.path.join('prototransfer/checkpoints/{}_{}_{}_{}_{}supp_{}query_{}bs'\
                                    .format(args.self_sup_loss,
                                            args.dataset, args.backbone,
                                            args.distance,
                                            args.train_support_shots,
                                            args.train_query_shots,
                                            args.batch_size))
        if args.n_images is not None:
            default_path += '_' + str(args.n_images) + 'images'
        if args.n_classes is not None:
            default_path += '_' + str(args.n_classes) + 'classes'
        save_path = args.save_path or default_path
        print('Save path is:', save_path)

        def save_checkpoint(state, is_best):
            if args.save:
                filename = save_path + '.pth.tar'
                torch.save(state, filename)
                if is_best:
                    shutil.copyfile(filename, save_path + '_best.pth.tar')

        # Load path
        if args.load_last:
            args.load_path = default_path + '.pth.tar'
        if args.load_best:
            args.load_path = default_path + '_best.pth.tar'

        # Load training state
        n_no_improvement = 0
        best_loss = np.inf
        best_accuracy = 0
        start_epoch = 0

        # Adjust patience
        if args.patience < 0:
            print('No early stopping!')
            args.patience = np.inf
        else:
            print('Early stopping with patience {} epochs'.format(args.patience))

    # Load checkpoint
    if args.load_path:
        try: # Cannot load CUDA trained models onto cpu directly
            checkpoint = torch.load(args.load_path)
        except:
            checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'))
        proto.encoder.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        if mode == 'train':
            n_no_improvement = checkpoint['n_no_improvement']
            best_loss = checkpoint['best_loss']
            best_accuracy = checkpoint['best_accuracy']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loaded checkpoint '{}' (epoch {})"
              .format(args.load_path, start_epoch))

    ##########################################
    # Define train and test functions
    ##########################################

    def train_epoch(model, dataloader, optimizer, scheduler):
        model.train()
        accuracies = []
        losses = []
        for iteration, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss, accuracy = model.forward(batch)
            accuracies.append(accuracy)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            if iteration == args.iterations-1:
                break
        return np.mean(losses), np.mean(accuracies)


    def eval_epoch(model, dataloader, iterations,
                   sup_finetune=False, progress_bar=False,
                   trainval=False):
        model.eval()
        losses = []
        accuracies = []
        # Keep original state to reset after finetuning (deepcopy required)
        if not trainval:
            original_encoder_state = copy.deepcopy(model.encoder.state_dict())

        # Define iterator
        episodes = enumerate(dataloader)
        if progress_bar:
            episodes = tqdm(episodes, unit='episodes',
                            total=iterations, initial=0, leave=False)

        # Perform evaulation episodes
        for iteration, episode in episodes:
            if sup_finetune:
                loss, accuracy = supervised_finetuning(model.encoder,
                                                        episode=episode,
                                                        inner_lr=args.sup_finetune_lr,
                                                        total_epoch=args.sup_finetune_epochs,
                                                        freeze_backbone=args.ft_freeze_backbone,
                                                        finetune_batch_norm=args.finetune_batch_norm,
                                                        device=device,
                                                        n_way=args.eval_ways)
                model.encoder.load_state_dict(original_encoder_state)
            elif trainval: # evaluating ProtoCLR loss/accuracy on train set
                with torch.no_grad():
                    loss, accuracy = model.forward(episode)
            else: 
                with torch.no_grad():
                    loss, accuracy = model.loss(episode, args.eval_ways)

            losses.append(loss.item())
            accuracies.append(accuracy)

            if iteration == iterations - 1:
                break

        conf_interval = st.t.interval(0.95, len(accuracies)-1, loc=np.mean(accuracies),
                                      scale=st.sem(accuracies))
        return np.mean(losses), np.mean(accuracies), np.std(accuracies), conf_interval

    def train(self_sup_loss, proto, trainloader,# valloader,
              optimizer, scheduler,
              n_no_improvement=0, best_loss=np.inf, best_accuracy=0,
              start_epoch=0):
        epochs = tqdm(range(start_epoch, args.epochs), unit='epochs',
                      total=args.epochs, initial=start_epoch)
        for epoch in epochs:
            # Train
            loss, accuracy = train_epoch(self_sup_loss, trainloader, optimizer, scheduler)
            # Validation
            #loss, accuracy, _, _ = eval_epoch(self_sup_loss, valloader, args.iterations)

            # Record best model, loss, accuracy
            best_epoch = accuracy > best_accuracy
            #best_epoch = loss < best_loss
            if best_epoch:# and epoch % 20 == 0:
                best_loss = loss
                best_accuracy = accuracy

                # Save checkpoint
                save_checkpoint({
                    'epoch': epoch + 1,
                    'n_no_improvement': n_no_improvement,
                    'model': proto.encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': loss,
                    'best_loss': best_loss,
                    'best_accuracy': best_accuracy,
                    'accuracy': accuracy,
                    'setup': args
                    }, best_epoch)

                n_no_improvement = 0

            else:
                n_no_improvement += 1

            # Update progress bar information
            epochs.set_description_str(desc='Epoch {}, loss {:.4f}, accuracy {:.4f}'\
                                       .format(epoch+1, loss, accuracy))

            # Early stopping
            if n_no_improvement > args.patience:
                print('Early stopping at epoch {}, because there was no\
                      improvement for {} epochs'.format(epoch, args.patience))
                break

        print('-------------- Best validation loss {:.4f} with accuracy {:.4f}'.format(best_loss, best_accuracy))

        best_checkpoint = torch.load(save_path + '_best.pth.tar')
        best_model = best_checkpoint['model']
        proto.encoder.load_state_dict(best_model)
        return proto

    ##########################################
    # Run training and evaluation
    ##########################################

    # Training
    if mode == 'train':
        print('Setting:')
        print(args, '\n')
        print('Training ...')
        best_model = train(self_sup_loss, proto, dataloader_train,# dataloader_val,
                           optimizer, scheduler, n_no_improvement,
                           best_loss, best_accuracy, start_epoch)
    elif mode == 'trainval':
        best_model = self_sup_loss
    else:
        best_model = proto

    # Evaluation
    if (mode == 'test') or (mode == 'trainval'):
        print('Evaluating ' + args.load_path + '...')
        test_loss, test_accuracy, test_accuracy_std, test_conf_interval \
                = eval_epoch(best_model, dataloader_test, args.test_iterations,
                             progress_bar=True, sup_finetune=args.sup_finetune,
                             trainval=mode=='trainval')

        print('Test loss {:.4f} and accuracy {:.2f} +- {:.2f}'.format(test_loss,
                                                                      test_accuracy*100,
                                                                      (test_conf_interval[1]-test_accuracy)*100))

