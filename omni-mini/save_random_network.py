import torch
import numpy as np
import shutil

from prototransfer.models import CNN_4Layer

def main():
    ##########################################
    # Setup
    ##########################################

    # Check whether GPU is available
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Used device:', device)

    # Define model
    ## Encoder
    encoder = CNN_4Layer(in_channels=3)
    encoder = encoder.to(device)

    # Save path
    save_path = 'prototransfer/checkpoints/random_init_conv4'
    print('Save path is:', save_path)

    def save_checkpoint(state):
        filename = save_path + '.pth.tar'
        torch.save(state, filename)

    # Save checkpoint
    save_checkpoint({
        'epoch': 0,
        'n_no_improvement': 0,
        'model': encoder.state_dict(),
        'optimizer': None,
        'scheduler': None,
        'loss': np.inf,
        'best_loss': np.inf,
        'best_accuracy': 0,
        'accuracy': 0,
        'setup': None
        })

if __name__ == '__main__':
    main()
