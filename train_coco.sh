export CUDA_VISIBLE_DEVICES=3
python scripts/dist_clip_coco.py \
       --config /home/student/minseo/WeCLIP/configs/coco_attn_reg.yaml \
       --match_ratio 0.7

python scripts/dist_clip_coco.py \
       --config /home/student/minseo/WeCLIP/configs/coco_attn_reg.yaml \
       --match_ratio 0.75

python scripts/dist_clip_coco.py \
       --config /home/student/minseo/WeCLIP/configs/coco_attn_reg.yaml \
       --match_ratio 0.8