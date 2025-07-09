#!/bin/bash

file=./scripts/train_voc.py
device_gpu=3
nproc_per_node=1
master_port=29733
exp_des=train_voc

for fuse_weight in 0.1 0.3 0.5 0.7 0.9 1
do
    CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file \
                                            --log_tag=$exp_des \
                                            --fuse_weight=$fuse_weight \
                                            --fuse_ver=2 \
                                            --extract_noun \
                                            --aug_first
done

for fuse_weight in 0.1 0.3 0.5 0.7 0.9 1
do
    CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file \
                                            --log_tag=$exp_des \
                                            --fuse_weight=$fuse_weight \
                                            --fuse_ver=2 \
                                            --extract_noun \
                                            --aug_first
done
