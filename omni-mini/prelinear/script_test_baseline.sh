#!/bin/bash
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 4  --train_aug --n_shot 1 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 8  --train_aug --n_shot 1 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 16  --train_aug --n_shot 1 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 32  --train_aug --n_shot 1 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 4  --train_aug --n_shot 5 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 8  --train_aug --n_shot 5 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 16  --train_aug --n_shot 5 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 32  --train_aug --n_shot 5 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 4  --train_aug --n_shot 20 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 8  --train_aug --n_shot 20 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 16  --train_aug --n_shot 20 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 32  --train_aug --n_shot 20 &

python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 4  --train_aug --n_shot 50 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 8  --train_aug --n_shot 50 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 16  --train_aug --n_shot 50 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_classes 32  --train_aug --n_shot 50 &

python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 2400  --train_aug --n_shot 1 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 4800  --train_aug --n_shot 1 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 9600  --train_aug --n_shot 1 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 19200  --train_aug --n_shot 1 &

python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 2400  --train_aug --n_shot 5 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 4800  --train_aug --n_shot 5 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 9600  --train_aug --n_shot 5 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 19200  --train_aug --n_shot 5 &

python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 2400  --train_aug --n_shot 20 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 4800  --train_aug --n_shot 20 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 9600  --train_aug --n_shot 20 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 19200  --train_aug --n_shot 20 &

python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 2400  --train_aug --n_shot 50 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 4800  --train_aug --n_shot 50 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 9600  --train_aug --n_shot 50 &
python ./test.py --dataset miniImagenet --model Conv4 --method baseline --limit_n_images 19200  --train_aug --n_shot 50 &
