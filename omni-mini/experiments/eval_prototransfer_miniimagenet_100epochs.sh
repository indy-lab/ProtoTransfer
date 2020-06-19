#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval ProtoCLR: mini-ImageNet         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

	python prototransfer/eval.py --dataset miniimagenet \
		--eval_ways 5 \
		--eval_support_shots 1 \
		--eval_query_shots 15 \
		--sup_finetune_epochs 100 \
		--finetune_batch_norm \
		--sup_finetune \
		--load_path prototransfer/checkpoints/protoclr/proto_cub_conv4_euclidean_1supp_3query_50bs_best.pth.tar

	echo This was: 5 classes, 1 shot

