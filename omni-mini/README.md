# Self-Supervised Prototypical Pre-training for Few-Shot Classification

## Code organisation
* [prototransfer](semifew): Contains the core code for our experiments.
* [setup](setup): Contains installation files.
* [experiments](experiments): Contains scripts to run the main experiments.

All commands and scripts should be executed from *this* top level directory.

## Main results
***To be filled with plots of main results***

## Setup
* This code has been tested on Ubuntu 18.04 with Python 3.7 and PyTorch 1.4.0.

### Install dependencies
#### via pip
```bash
cd omni-mini/setup
pip install -r requirements.txt
```

#### via conda
```bash
conda create -n <environment_name>
conda activate <environment_name>
conda install setuptools scipy numpy Pillow h5py tqdm requests
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
pip install torchmeta
```

#### via docker
```bash
cd omni-mini/setup
docker build -t <imagename> .
```
Depending on your docker setup, docker might not be able to access your host network. In that case add `--network=host`.

### Download datasets

The datasets by default will be downloaded in the `ProtoTransfer/few_data/` folder, which can be adjusted in the `configs.py` files inside the `cdfsl-benchmark` and `omni-mini` folders.

#### mini-ImageNet, Omniglot & CUB
Omniglot and CUB can be downloaded via the script `python prototransfer/load_data.py --dataset <dataset_name>`.

`python prototransfer/load_data.py --dataset miniimagnet`

`python prototransfer/load_data.py --dataset omniglot`

`python prototransfer/load_data.py --dataset cub`

## Running experiments
#### For training:
Use `python prototransfer/train.py --<args>`, where the following arguments are used to run our reported experiments:
* `--train_support_shots` for the number of support images
* `--train_query_shots` for the number of query images
* `--no_aug_support` to not augment the support
* `--no_aug_query` to not augment the query
* `--n_images` to limit the number of images to load for training
* `--n_classes` to limit the number of classes to load for training

This will save a checkpoint in `prototransfer/checkpoints/ `

For example *to train the ProtoCLR model described in our paper* with 1 support and 3 query examples run:
```bash
python prototransfer/train.py --dataset miniimagenet \
	  --train_support_shots 1 \
	  --train_query_shots 3 \
    	  --no_aug_support
```

To train the models in our ablation studies simply add `--n_images <int>` or `--n_classes <int>`.

*Many more argumentes can be found in `prototransfer/train.py`*

#### For evaluation:
Use `python prototransfer/eval.py --<args>`, where the following arguments are used to evaluate our reported experiments:
* `--eval_ways` for the number of ways during evaluation
* `--eval_support_shots` for the number of support images
* `--eval_query_shots` for the number of query images
* `--sup_finetune` to perform finetune on target domain
* `--ft_freeze_backbone` to not finetune the embedding network 
* `--load_path` to specify the exact path from which to load a pre-trained checkpoint

To *evaluate our proposed ProtoTransfer model* trained with 1 support and 3 query examples in a 5-way 1-shot setting, run:
```
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots 1 \
	--eval_query_shots 15 \
	--sup_finetune \
	--ft_freeze_backbone \
	--load_path prototransfer/checkpoints/protoclr/protoclr_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar
```

Pre-trained models for all our experiments can be found in the different folders within `prototransfer/checkpoints/`.
* ablation_n_images: contains the checkpoints for reducing the number of images or classes.
	* example name for reduced \# images: `proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_1200images_best.pth.tar`
	* example name for reduced \# classes: `proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_2classes_best.pth.tar`
* protoclr: contains the different protoclr checkpoints with different numbers of queries etc.
	* exmaple name for mini-ImageNet: `protoclr_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar`
	* example name for Omniglot: `proto_omniglot_conv4_euclidean_1supp_3query_50sbs_best.pth.tar`
* umtra: contains the umtra checkpoints.
	* for mini-ImagNet: `umtra_miniimagenet_conv4_euclidean_1supp_1query_5bs_best.pth.tar`
	* for Omniglot: `umtra_omniglot_conv4_euclidean_1supp_1query_5bs_best.pth.tar`


*Many more argumentes can be found in `prototransfer/eval.py`*

#### Bash files
Alternatively, we also provide bash scripts for most used setups. 
```bash
experiments/<experiment_name>.sh
```

For example to train the ProtoCLR model described in our paper with 1 support and 3 query examples run:
```
bash experiments/protoclr_miniimagenet.sh
```

To evaluate the same model using ProtoTune as described in our paper run:
```
bash experiments/eval_prototransfer_miniimagenet.sh
```
where `experiments/eval_prototransfer_miniimagenet.sh` will evaluate also the ablation study for reducing the number of images or classes. Please refer to the bash file for details, but here is an excerpt:

```bash
# Standard eval
echo ++++++++++ Standard ProtoTrasnfer ++++++++++++++++
for n_shot in 1 5 20 50
do
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--sup_finetune \
	--ft_freeze_backbone \
	--load_path prototransfer/checkpoints/protoclr/protoclr_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar
echo This was: All images, ${n_shot} shot

...
```


## Citation
If you use this code, please cite our paper:

```
@inproceedings{}
```

