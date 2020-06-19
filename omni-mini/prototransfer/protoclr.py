"""Contrastive loss forumulation"""
import torch
import torch.nn as nn

from prototransfer.utils_proto import (prototypical_loss,
                                       get_prototypes)

class ProtoCLR(nn.Module):
    """Calculate the ProtoCLR loss on a batch of images.

    Parameters:
        - encoder: The encoder network.
        - device: The device to use.
        - n_support: The number of support shots.
        - n_query: The number of query shots.
    """
    def __init__(self, encoder, device='cpu', n_support=1, n_query=1,
                 distance='euclidean'):
        super(ProtoCLR, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        
        self.encoder = encoder
        self.device = device
        self.distance = distance

    def forward(self, batch):
        # Treat the first dim as way, the second as shots
        # and use the first shot as support (like 1-shot setting)
        # the remaining shots as query
        data = batch['data'].to(self.device) # [batch_size x ways x shots x image_dim]
        data = data.unsqueeze(0) #TODO remove when multi-batching is supported
        # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
        batch_size = data.size(0)
        ways = data.size(1)
        
        # Divide into support and query shots 
        x_support = data[:,:,:self.n_support]
        x_support = x_support.reshape((batch_size, ways * self.n_support, *x_support.shape[-3:])) # e.g. [1,50*n_support,*(3,84,84)]
        x_query = data[:,:,self.n_support:]
        x_query = x_query.reshape((batch_size, ways * self.n_query, *x_query.shape[-3:])) # e.g. [1,50*n_query,*(3,84,84)]
        
        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(0).unsqueeze(2) # batch and shot dim
        y_query = y_query.repeat(batch_size, 1, self.n_query)
        y_query = y_query.view(batch_size, -1).to(self.device)
        
        y_support = torch.arange(ways).unsqueeze(0).unsqueeze(2) # batch and shot dim
        y_support = y_support.repeat(batch_size, 1, self.n_support)
        y_support = y_support.view(batch_size, -1).to(self.device)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1) # e.g. [1,50*(n_support+n_query),*(3,84,84)]
        z = self.encoder.forward(x)
        z_support = z[:,:ways * self.n_support] # e.g. [1,50*n_support,*(3,84,84)]
        z_query = z[:,ways * self.n_support:] # e.g. [1,50*n_query,*(3,84,84)]

        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)

        return loss, accuracy

