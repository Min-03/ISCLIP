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

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")


def make_caption(cfg):
    
    # train_dataset = voc.VOC12ClsDataset(
    #     root_dir=cfg.dataset.root_dir,
    #     name_list_dir=cfg.dataset.name_list_dir,
    #     split=cfg.train.split,
    #     stage='train',
    #     aug=True,
    #     resize_range=cfg.dataset.resize_range,
    #     rescale_range=cfg.dataset.rescale_range,
    #     crop_size=cfg.dataset.crop_size,
    #     img_fliplr=True,
    #     ignore_index=cfg.dataset.ignore_index,
    #     num_classes=cfg.dataset.num_classes,
    # )
    
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.train.samples_per_gpu,
    #                           shuffle=True,
    #                           num_workers=10,
    #                           pin_memory=False,
    #                           drop_last=True,
    #                           prefetch_factor=4)
    
    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).cuda()
    
    # name2cap = {}
    # img_dir = "/data/dataset/VOC2012/JPEGImages"
    
    # for img_names, inputs, cls_labels, img_box in tqdm(train_loader):
    #     img_list = [Image.open(os.path.join(img_dir, img_name+'.jpg')).convert('RGB') for img_name in img_names]
    #     inputs = processor(images=img_list, return_tensors="pt", do_rescale=True).to(model.device)
    #     out = model.generate(**inputs)

    #     caption_list = processor.batch_decode(out, skip_special_tokens=True)
    #     for name, caption in zip(img_names, caption_list):
    #         name2cap[name] = caption
        
    caption_file_dir = "/data/dataset/VOC2012/train_cap.pickle"
    # with open(caption_file_dir, "wb") as fw:
    #     pickle.dump(name2cap, fw)
    with open(caption_file_dir, "rb") as fr:
        name2cap = pickle.load(fr)

    save_dir = "/data/dataset/VOC2012/ParsedCaptions"
    grammar = "ADJ_NOUN: {<DT>?<JJ>*<NN|NNS>+}"
    chunk_parser = RegexpParser(grammar)
    problems = ["a white and brown puppy\n", "a boy holding two fish\n", "two sheep standing next to each other\n", "a pelican with its young\n"]
    
    for name, caption in tqdm(name2cap.items()):
        words = word_tokenize(caption)
        tagged_words = pos_tag(words)

        tree = chunk_parser.parse(tagged_words)
        if caption in problems:
            tree.pretty_print()          # Print tree as text

        phrases = [" ".join(word for word, tag in subtree.leaves()) for subtree in tree.subtrees() if subtree.label() == "ADJ_NOUN"]
        # if len(phrases) == 0:
        #     print(name, caption)
        # file_name = os.path.join(save_dir, f"{name}.txt")
        # with open(file_name, "w", encoding="utf-8") as f:

            
        #     parsed_caption = "#".join(phrases)
        #     # f.write(parsed_caption)
    
    
        
        
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    make_caption(cfg=cfg)
