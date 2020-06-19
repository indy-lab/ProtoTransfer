#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++      Baseline: Omniglot              +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

python prototransfer/train.py --dataset omniglot \
	--method baseline \