import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append(".")
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import voc
from tqdm import tqdm
import json
from omegaconf import OmegaConf

from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from clip.clip_text import class_names
import pickle

def make_caption(cfg):
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)

    model_name = "Salesforce/instructblip-vicuna-7b"
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

    img_dir = "/data/dataset/VOC2012/JPEGImages"
    caption_file_dir = "/data/dataset/VOC2012/SpecificCaption2"
    
    for img_names, inputs, cls_labels, img_box in tqdm(train_loader):
        img = Image.open(os.path.join(img_dir, img_names[0] + '.jpg')).convert('RGB')

        present_cls = torch.nonzero(cls_labels[0])
        
        present_cls_name = [class_names[idx] for idx in present_cls]

        instructions = [f"What does {cls} look like?" for cls in present_cls_name]

        captions = {}
        for idx, instruction in zip(present_cls, instructions):
            inputs = processor(img, instruction, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            output = model.generate(**inputs, 
                                    max_new_tokens=50,
                                    )
            caption = processor.decode(output[0], skip_special_tokens=True)
            caption = caption[len(instruction):]
            captions[idx] = caption
            
        caption_dir = os.path.join(caption_file_dir, f"{img_names[0]}.pickle")
        with open(caption_dir, "wb") as fw:
            pickle.dump(captions, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='/home/student/minseo/ISCLIP/configs/voc_attn_reg.yaml',
                        type=str,
                        help="config")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    make_caption(cfg=cfg)