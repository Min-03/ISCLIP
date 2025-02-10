export CUDA_VISIBLE_DEVICES=3
python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --match_ratio 0.65 \
       --num_workers 8

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --match_ratio 0.70 \
       --num_workers 8

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --match_ratio 0.75 \
       --num_workers 8

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --match_ratio 0.80 \
       --num_workers 8

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --refine_with_img \
       --num_workers 8