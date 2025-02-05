import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

from datasets import coco
from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_model.model_attn_aff_voc import WeCLIP
from pycocotools.coco import COCO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/home/student/minseo/WeCLIP/configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")

def save_caption(img_name, caption, file_path="captions.json"):
    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Add the new entry
    data[img_name] = caption

    # Save back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved: {img_name} -> {caption}")

def get_contrast_loss(org_prompts, refined_prompts):
    """
    Args:
        org_prompts (F+B, H): original class prompts
        refined_prompts (bs, F+B, H) refined class prompts

    Returns:
        matching loss for two prompts
    """
    B, C, H = refined_prompts.shape
    
    org_prompts = F.normalize(org_prompts, dim=-1)
    refined_prompts = F.normalize(refined_prompts, dim=-1)
    logits = torch.matmul(refined_prompts, org_prompts.T)
    gt = torch.eye(C).cuda().unsqueeze(0).expand(B, C, C)
    return F.cross_entropy(logits, gt, reduction='none')

def train(cfg):
    annotation_file_dir = os.path.join(cfg.dataset.root_dir, "train_cap.pickle")
    
    train_dataset = voc.VOC12CapClsDataset(
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
    
    WeCLIP_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda'
    )
    
    train_loader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)
    
    WeCLIP_model.cuda()
    
    train_loader_iter = iter(train_loader)
    
    
    for n_iter in range(5):        
        try:
            img_names, inputs, cls_labels, img_box, captions = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_names, inputs, cls_labels, img_box, captions = next(train_loader_iter)
        
        segs, cam, attn_pred, prompts = WeCLIP_model(inputs.cuda(), img_names, captions, requires_prompt=True)
        print(get_contrast_loss(prompts[0], prompts[1]).shape)

    
    
if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg=cfg)
