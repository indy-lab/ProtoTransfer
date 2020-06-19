#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval ProtoCLR: CUB2mini-Imagenet     +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

echo 'ProtoCLR'
for n_shot in 1 5 20 50
do
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--load_path prototransfer/checkpoints/proto_cub_conv4_euclidean_1supp_3query_50bs_best.pth.tar

echo This was: ProtoCLR ${n_shot} shot
done

echo 'ProtoTransfer'
for n_shot in 1 5 20 50
do
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--sup_finetune \
	--ft_freeze_backbone \
	--load_path prototransfer/checkpoints/proto_cub_conv4_euclidean_1supp_3query_50bs_best.pth.tar

echo This was: ProtoTrasnfer ${n_shot} shot
done
