export CUDA_VISIBLE_DEVICES=3

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/ISCLIP/configs/voc_attn_reg.yaml \
       --fuse_ver 4 \
       --fuse_mode cls_txt \
       --num_workers 4

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/ISCLIP/configs/voc_attn_reg.yaml \
       --fuse_ver 4 \
       --fuse_mode cls_txt \
       --refine_bg \
       --num_workers 4