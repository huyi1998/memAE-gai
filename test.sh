#! /bin/bash
# sed -i 's/\r$//' test.sh 解决window写shell到linux换行符的问题
for ckptstep in 50 60 65
do
python Testing.py --dataset_type factory3 --dataset_path ../dataset/ --version 2 --EntropyLossWeight 0 --lr 2e-4 --exp_dir log/ --ckpt_step $ckptstep
done