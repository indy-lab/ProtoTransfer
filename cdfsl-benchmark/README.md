# Code for our experiments on the Cross-Domain Few-Shot Learning (CDFSL) Benchmark

## Code organisation
* `train.py`: File to call for (pre-)training
* `finetune.py`: File to call for testing and optional fine-tuning
* `io_utils.py`: File detailing training/testing arguments
* `configs.py`: Configure location of data, as detailed in the [CDFSL benchmark repository](https://github.com/IBM/cdfsl-benchmark)

## Setup
* This code has been tested on Ubuntu 18.04 with Python 3.7 and PyTorch 1.4.0.
* For installation instructions consult the original [CDFSL benchmark repository](https://github.com/IBM/cdfsl-benchmark).

## Running experiments
### Pre-training on mini-ImageNet (Optional)

This step is optional as pre-trained models used in the paper are provided.

#### Pre-train with ProtoTransfer on mini-ImageNet:

`python train.py --dataset miniImagenet --method prototransfer --stop_epoch 600 --n_support 1 --n_query 3 --no_aug_support`

#### Pre-train with UMTRA-ProtoNet on mini-ImageNet:

`python train.py --dataset miniImagenet --method umtra --stop_epoch 600 --batch_size 5 --n_support 1 --n_query 1 --no_aug_support`

### Fine-Tuning on the 4 CDFSL datasets

The `finetune.py` script loads a pre-trained model (see above) and automatically tests on all 4 datasets: ISIC, CropDiseases, EuroSAT and ChestXray. All testing experiments use `5-way` (classes) but can be adjusted in terms of `K-shots` by setting `--n_test_shot K`, see below for examples from the paper.

#### ProtoTransfer (ProtoCLR + ProtoTune)

`python finetune.py --model ResNet10 --method prototransfer --train_aug --n_support 1 --n_query 3 --no_aug_support --adaptation --n_test_shot 5`

#### ProtoCLR + ProtoNet

`python finetune.py --model ResNet10 --method prototransfer --train_aug --n_support 1 --n_query 3 --no_aug_support --n_test_shot 5`

#### UMTRA + ProtoTune

`python finetune.py --model ResNet10 --method umtra --train_aug --n_support 1 --n_query 1 --no_aug_support --adaptation --batch_size 5 --n_test_shot 5`

#### UMTRA + ProtoNet

`python finetune.py --model ResNet10 --method umtra --train_aug --n_support 1 --n_query 1 --no_aug_support --batch_size 5 --n_test_shot 5`

## Instructions:
All command line arguments can be found in `io_utils.py`

### Self-supervised Pre-Training
Training is done via `python train.py`
* `--method prototclr` or `--method umtra` changes which method is used. Impoortant to note is that this also determines how the checkpoints are saved. Default is `protoclr`.
* `--save_freq <int>` will change the checkpoint saving frequency. Default is `25`.
* `--dataset <dataset_name>` will change what dataset will be used for training. Default is `miniImagenet`.
* `--n_support 1` and `--n_query 1` allow for flexible number of support/query & augmentations. Defaults is 1 for both.
* `--no_aug_support` and `--no_aug_query` are used when you DON’t want an augmentation on either the support or query examples during training
* `--stop_epoch <epoch number>` determines the number of training epochs. Default is 400.

### Transfer learning and testing
Testing and optional fine-tuning is done via `finetune.py`
* `--method prototransfer` or `--method umtra` changes for which method the checkpoint will be loaded. Default is `prototransfer`.
* `--save_iter <int>` will change the checkpoint to load. Default is the last checkpoint.
* `--adaptation` will enable fine-tuning on the target domain. Without it a ProtoNet distance classifier is used. With `--adaptation` a linear classifier is trained on top.
* `--freeze_backbone` will freeze the backbone during finetuning. Only has an effect in combination with `--adaptation`.
* `--proto_init` will initialise the linear classifier with weights and biases based on distances to prototypes. This is in line with the interpretation of ProtoNets as linear classifiers. Only has an effect in combination with `--adaptation`.
* `--ft_steps <n_steps>` controls the number of finetuning epochs. Default is `15`
* `--lr_rate <some float>` controls the inner learning rate for fine-tuning. Default is `0.001`

## Changes to the orgininal CDFSL benchmark repository
[Original CDFSL benchmark repository here](https://github.com/IBM/cdfsl-benchmark)
* Most experiments from the CDFSL-paper will not run anymore, because of changed datalaoders
* Datasets do not load into RAM anymore, but images are loaded on the fly (various files /datasets/\*few_shot.py)
* Only minimally necessary code changes were performed to keep compatibility
