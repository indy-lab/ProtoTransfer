#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval ProtoCLR: Omniglot              +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for way in 5 20
do
	for shot in 1 5
	do
		python prototransfer/eval.py --dataset omniglot \
			--eval_ways ${way} \
			--eval_support_shots ${shot} \
			--eval_query_shots 15 \
			--load_path prototransfer/checkpoints/protoclr/proto_omniglot_conv4_euclidean_1supp_3query_50bs_best.pth.tar
		echo This was ${way} way, ${shot} shot
	done
done
