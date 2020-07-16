# Self-Supervised Prototypical Transfer Learning for Few-Shot Classification
This repository contains the reference source code and pre-trained models (ready for evaluation) for our paper [*Self-Supervised Prototypical Transfer Learning for Few-Shot Classification*](https://arxiv.org/abs/2006.11325).

Part of this work has been presented at the [ICML 2020 Workshop on Automated Machine Learning](https://sites.google.com/view/automl2020/home).

![ProtoTransfer method illustration](https://github.com/indy-lab/ProtoTransfer/raw/master/method_illustration.png)
## Structure

### `omni-mini/`
Contains instructions and all runnable code for ProtoTransfer & UMTRA for our Omniglot and mini-ImageNet experiments

### `cdfsl-benchmark/`
Contains instructions, all runnable code and pre-trained models for ProtoTransfer & UMTRA for our CDFSL benchmark experiments

## Setup
For setting up a Python environment to run our experiments, please refer to [omni-mini/setup](omni-mini/setup). The dataset setups can be found in `omni-mini` and `cdfsl-benchmark`.

## Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@article{medina2020selfsupervised,
    title="{Self-Supervised Prototypical Transfer Learning for Few-Shot Classification}",
    author={Carlos Medina and Arnout Devos and Matthias Grossglauser},
    journal={arXiv preprint arXiv:2006.11325},
    year={2020}
}
```
