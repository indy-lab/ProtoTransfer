#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++      ProtoCLR: Miniimagenet            +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

python prototransfer/train.py --dataset miniimagenet \
	--train_support_shots 1 \
	--train_query_shots 3 \
        --no_aug_support \
        --n_classes 1
