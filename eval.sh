export CUDA_VISIBLE_DEVICES=3

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/base1.pth

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/base2.pth

python test_msc_flip_voc.py \
       --model_path /data/minseo/ISCLIP/base3.pth