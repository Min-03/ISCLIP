export CUDA_VISIBLE_DEVICES=3
python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --fuse_ver 4 \
       --fuse_mode txt \
       --num_workers 8

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --fuse_ver 4 \
       --fuse_mode txt_parsed \
       --match_ratio 0.70 \
       --num_workers 8

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --fuse_ver 4 \
       --fuse_mode img \
       --num_workers 8

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --fuse_ver 4 \
       --fuse_mode txt \
       --num_workers 8 \
       --refine_always

# python scripts/dist_clip_voc.py \
#        --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
#        --fuse_ver 2 \
#        --fuse_mode txt \
#        --num_workers 8

# python scripts/dist_clip_voc.py \
#        --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
#        --fuse_ver 2 \
#        --fuse_mode txt_parsed \
#        --match_ratio 0.70 \
#        --num_workers 8

# python scripts/dist_clip_voc.py \
#        --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
#        --fuse_ver 2 \
#        --fuse_mode img \
#        --num_workers 8

# python scripts/dist_clip_voc.py \
#        --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
#        --fuse_ver 3 \
#        --fuse_mode txt \
#        --num_workers 8

# python scripts/dist_clip_voc.py \
#        --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
#        --fuse_ver 3 \
#        --fuse_mode txt_parsed \
#        --match_ratio 0.70 \
#        --num_workers 8

# python scripts/dist_clip_voc.py \
#        --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
#        --fuse_ver 3 \
#        --fuse_mode img \
#        --num_workers 8