#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval ProtoCLR: Omniglot              +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

python prototransfer/eval.py --dataset omniglot \
	--eval_ways 20 \
        --eval_support_shots 5 \
        --eval_query_shots 15 \
	--sup_finetune \
	--ft_freeze_backbone \
	--load_path prototransfer/checkpoints/protoclr/protoclr_omniglot_conv4_euclidean_1supp_3query_50bs_best.pth.tar
