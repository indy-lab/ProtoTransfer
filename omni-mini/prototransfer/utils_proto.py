"""Adapted from
1) https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/utils/prototype.py
2) https://github.com/tristandeleu/pytorch-meta/blob/master/examples/protonet/utils.py
"""

import torch
import torch.nn.functional as F

def euclidean_distance(x, y):
    """x, y have shapes (batch_size, num_examples, embedding_size)."""
    return torch.sum((x.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=-1)

def cosine_similarity(x, y):
    """x, y have shapes (batch_size, num_examples, embedding_size)."""
    # Compute dot products x_i.T y_i (numerator)
    dot_similarity = torch.bmm(x, y.permute(0, 2, 1))
    
    # Compute l2 norms ||x_i|| * ||y_i|| (denominator)
    x_norm = x.norm(p=2, dim=-1, keepdim=True)
    y_norm = y.norm(p=2, dim=-1, keepdim=True)
    norms = torch.bmm(x_norm, y_norm.permute(0, 2, 1)) + 1e-08 # avoid 0 division

    return dot_similarity / norms

def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support 
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor 
        has shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has 
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def prototypical_loss(prototypes, embeddings, targets, 
                      distance='euclidean', **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical 
    network, on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(batch_size, num_classes, embedding_size)`.

    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(batch_size, num_examples)`.

    distance : `String`
        The distance measure to be used: 'eucliden' or 'cosine'

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    if distance == 'euclidean':
        squared_distances = euclidean_distance(prototypes, embeddings)
        loss = F.cross_entropy(-squared_distances, targets, **kwargs)
        _, predictions = torch.min(squared_distances, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    elif distance == 'cosine':
        cosine_similarities = cosine_similarity(prototypes, embeddings)
        loss = F.cross_entropy(cosine_similarities, targets, **kwargs)
        _, predictions = torch.max(cosine_similarities, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    else:
        raise ValueError('Distance must be "euclidean" or "cosine"')
    return loss, accuracy.item()
