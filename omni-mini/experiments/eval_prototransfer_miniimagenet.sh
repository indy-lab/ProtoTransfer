#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval ProtoCLR: mini-ImageNet         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

# Standard eval
echo ++++++++++ Standard ProtoTrasnfer ++++++++++++++++
for n_shot in 1 5 20 50
do
python prototransfer/eval.py --dataset miniimagenet \
	--eval_ways 5 \
	--eval_support_shots ${n_shot} \
	--eval_query_shots 15 \
	--sup_finetune \
	--ft_freeze_backbone \
	--load_path prototransfer/checkpoints/protoclr/protoclr_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar
echo This was: All images, ${n_shot} shot

done

# N classes
echo ++++++++++++++++++++++++++ Varying the number of clases ++++++++++++++++++++++++++++++
for n_class in 1 2 4 8 16 32
do
    echo ------------------- ${n_class} classes --------------------

    for n_shot in 1 5 20 50
    do
	python prototransfer/eval.py --dataset miniimagenet \
		--eval_ways 5 \
		--eval_support_shots ${n_shot} \
		--eval_query_shots 15 \
		--sup_finetune \
		--ft_freeze_backbone \
		--load_path prototransfer/checkpoints/proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_${n_class}classes_best.pth.tar

	echo This was: ${n_class} classes, ${n_shot} shot
    done
done


echo ++++++++++++++++++++++++++ Varying the number of images ++++++++++++++++++++++++++++++
for n_images in 600 1200 2400 4800 9600 19200
do
    echo ------------------- ${n_images} images --------------------

    for n_shot in 1 5 20 50
    do
	python prototransfer/eval.py --dataset miniimagenet \
		--eval_ways 5 \
		--eval_support_shots ${n_shot} \
		--eval_query_shots 15 \
		--sup_finetune \
		--ft_freeze_backbone \
		--load_path prototransfer/checkpoints/proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_${n_images}images_best.pth.tar

	echo This was: ${n_images} images, ${n_shot} shot
    done
done
