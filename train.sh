export CUDA_VISIBLE_DEVICES=3
python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --m_weight 0.1

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --m_weight 0.05

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --m_weight 0.01

python scripts/dist_clip_voc.py \
       --config /home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml \
       --m_weight 0