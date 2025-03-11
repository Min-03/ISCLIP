export CUDA_VISIBLE_DEVICES=3

python test_msc_flip_voc.py \
       --model_path /data/minseo/WECLIP/pascal/ver7.pth \
       --fuse_ver 4 \
       --fuse_mode cls_txt \
       --work_dir /data/dataset/VOC2012/infer_results