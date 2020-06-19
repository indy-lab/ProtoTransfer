#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++      ProtoCLR: CUB                   +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

python prototransfer/train.py --dataset cub \
	--train_support_shots 1 \
	--train_query_shots 3 \
        --no_aug_support
