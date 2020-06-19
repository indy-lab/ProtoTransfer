#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval Random: CUB                     +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for n_shot in 1 5 20
do
python prototransfer/eval.py --dataset cub \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--sup_finetune \
	--ft_freeze_backbone \
	--load_path prototransfer/checkpoints/random_init_conv4.pth.tar

echo This was: ${n_shot} shot
done
