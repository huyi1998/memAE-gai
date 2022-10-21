#!/bin/bash
trap "exit" INT
datatype=factory2
datapath=../dataset/
version=11
ckptstep=40
expdir=log/

python Testing.py --dataset_type $datatype --dataset_path $datapath --version $version --EntropyLossWeight 0.0002 --lr 1e-4 --exp_dir $expdir --ckpt_step $ckptstep

# python Testing.py --dataset_type factory3 --dataset_path ../dataset/ --version 1 --EntropyLossWeight 0 --lr 2e-4 --exp_dir log/ --ckpt_step 60










