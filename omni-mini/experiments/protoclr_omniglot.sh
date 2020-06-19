#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++      ProtoCLR: Omniglot              +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

python prototransfer/train.py --dataset omniglot \
	--train_support_shots 1 \
	--train_query_shots 3 \
	--merge_train_val \
	--no_aug_support 
