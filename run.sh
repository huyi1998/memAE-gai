#!/bin/bash
datatype=factory2
datapath=../dataset/
expdir=log/
batchsize=10
epochs=70
version=4

python Train.py --dataset_path $datapath --dataset_type $datatype --version 0 --EntropyLossWeight 0 --lr 1e-4 --exp_dir $expdir --batch_size $batchsize --epochs $epochs --version $version







