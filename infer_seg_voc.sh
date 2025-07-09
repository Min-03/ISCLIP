#!/bin/bash

file=./tools/infer_seg_voc.py
inferset=val
crf=true

prefix="/data/minseo/ExCEL"
cpt_list=("weight_1.pth" "weight_3.pth" "weight_5.pth" "weight_9.pth")

# cpt=/data/minseo/ExCEL/final.pth

for name in ${cpt_list[@]}; do
    full_name="$prefix/$name"
    python $file --model_path $full_name --infer_set $inferset --crf_post $crf
done