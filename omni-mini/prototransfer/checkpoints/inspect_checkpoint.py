import torch
from torch import load
import os

def inspect_folder():
    print('Found checkpoint folders:')
    dirs = os.listdir()
    dirs = sorted([d for d in dirs if os.path.isdir(d)])
    for i, d in enumerate(dirs):
        print('[{}] '.format(i) + d)
    print('[{}] exit'.format(len(dirs)))

    folder = input('Please choose a folder number between 0 and {}:\
                   '.format(len(dirs)-1))
    folder = int(folder)

    if folder == len(dirs):
        return
    else:
        folder = dirs[folder]
        inspect_checkpoint(folder)

def inspect_checkpoint(folder):
    print('Found checkpoints ...')
    ckpts = os.listdir(folder)
    ckpts = sorted([c for c in ckpts if c.endswith('.pth.tar')])
    for i, c in enumerate(ckpts):
        print('[{}] '.format(i) + c)
    print('[{}] go back to other folders'.format(len(ckpts)))

    checkpoint = input('Please choose a checkpoint number between 0 and {}:\
                       '.format(len(ckpts)-1))
    checkpoint = int(checkpoint)
    if checkpoint == len(ckpts):
        inspect_folder()
    else:
        checkpoint = ckpts[checkpoint]
        try:
            checkpoint = load(os.path.join(folder, checkpoint))
        except:
            checkpoint = load(os.path.join(folder, checkpoint),
                             map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        try:
            setup = checkpoint['setup']
            print(setup)
            print()
        except:
            pass
        print('Validation loss {} and accuracy {} in epoch {}'\
              .format(loss, accuracy, epoch))
        print('----------- Press enter to continue -----------')
        input()
        inspect_checkpoint(folder)

if __name__ == '__main__':
    inspect_folder()
