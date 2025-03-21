export CUDA_VISIBLE_DEVICES=3

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --fuse_ver 1 \
       --fuse_mode txt \
       --aug_ver 3 \
       --debug \
       --resume /data/minseo/ISCLIP/base_aug3.pth