export CUDA_VISIBLE_DEVICES=3

python scripts/dist_clip_voc.py \
       --config configs/voc_attn_reg.yaml \
       --cap_dir /data/dataset/VOC2012/Cap \
       --train_mode no_cap

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/temp.pth \
       --work_dir /data/dataset/VOC2012/infer_results

python scripts/dist_clip_voc.py \
       --config configs/voc_attn_reg.yaml \
       --cap_dir /data/dataset/VOC2012/Cap \
       --train_mode no_cap

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/temp.pth \
       --work_dir /data/dataset/VOC2012/infer_results



python scripts/dist_clip_voc.py \
       --config configs/voc_attn_reg.yaml \
       --cap_dir /data/dataset/VOC2012/Cap \
       --fuse_weight 0.1

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/temp.pth \
       --work_dir /data/dataset/VOC2012/infer_results

python scripts/dist_clip_voc.py \
       --config configs/voc_attn_reg.yaml \
       --cap_dir /data/dataset/VOC2012/Cap \
       --fuse_weight 0.1

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/temp.pth \
       --work_dir /data/dataset/VOC2012/infer_results



python scripts/dist_clip_voc.py \
       --config configs/voc_attn_reg.yaml \
       --cap_dir /data/dataset/VOC2012/Cap \
       --fuse_weight 0.1 \
       --refine_cam

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/temp.pth \
       --work_dir /data/dataset/VOC2012/infer_results

python scripts/dist_clip_voc.py \
       --config configs/voc_attn_reg.yaml \
       --cap_dir /data/dataset/VOC2012/Cap \
       --fuse_weight 0.1 \
       --refine_cam

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/temp.pth \
       --work_dir /data/dataset/VOC2012/infer_results
