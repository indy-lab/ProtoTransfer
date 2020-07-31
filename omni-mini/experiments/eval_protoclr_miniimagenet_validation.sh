#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval ProtoCLR: mini-ImageNet         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

echo '----------------- ProtoTransfer ---------------------------'
for n_shot in 1 5 20 50
do
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--sup_finetune \
	--ft_freeze_backbone \
	--mode val \
	--load_path prototransfer/checkpoints/protoclr/proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar

echo This was: ProtoTrasnfer, ${n_shot} shot
done


echo '----------------- PROTOCLR ---------------------------'
for n_query in 1 3 5 10
do
    echo ------------------- ${n_query} queries --------------------

    for n_shot in 1 5 20 50
    do
	python prototransfer/eval.py --dataset miniimagenet \
		--eval_ways 5 \
		--eval_support_shots ${n_shot} \
		--eval_query_shots 15 \
		--mode val \
		--load_path prototransfer/checkpoints/protoclr/proto_miniimagenet_conv4_euclidean_1supp_${n_query}query_50bs_best.pth.tar

	echo This was: ${n_query} classes, ${n_shot} shot
    done
done

echo '----------------- UMTRA ---------------------------'
for n_shot in 1 5 20 50
do
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--mode val \
	--load_path prototransfer/checkpoints/umtra/umtra_miniimagenet_conv4_euclidean_1supp_1query_5bs_best.pth.tar

echo This was: UMTRA, ${n_shot} shot
done

echo '----------------- Random network ---------------------------'
for n_shot in 1 5 20 50
do
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--mode val \
	--load_path prototransfer/checkpoints/random_init_conv4.pth.tar

echo This was: Random Init, ${n_shot} shot
done

