#!/bin/bash
echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++ Eval ProtoCLR: mini-ImageNet         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

echo ------------------- ProtoCLR --------------------
for n_query in 1 3 5 10
do
    echo ------------------- ${n_query} queries --------------------

    python prototransfer/eval.py --dataset miniimagenet \
	--mode trainval \
	--load_path prototransfer/checkpoints/protoclr/protoclr_miniimagenet_conv4_euclidean_1supp_${n_query}query_50bs_best.pth.tar

   echo This was: ProtoCLR ${n_query} queries
done


echo ------------------- UMTRA --------------------
python prototransfer/eval.py --dataset miniimagenet \
	--mode trainval \
	--load_path prototransfer/checkpoints/umtra/umtra_miniimagenet_conv4_euclidean_1supp_1query_5bs_best.pth.tar

echo This was: UMTRA

echo ------------------- Random Network --------------------
python prototransfer/eval.py --dataset miniimagenet \
	--mode trainval \
	--load_path prototransfer/checkpoints/random_init_conv4.pth.tar

echo This was: UMTRA

