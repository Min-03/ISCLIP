
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import os
from torchvision.transforms import Compose, Normalize
from .decoder.TransDecoder import DecoderTransformer
import clip
from datasets.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY,new_class_names_coco, BACKGROUND_CATEGORY_COCO
from .load_attr import attr_aggregate
import os
import pickle
from utils.nlputils import extract_noun_phrase
import spacy

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


class ExCEL_model(nn.Module):
    def __init__(self,  clip_model=None, embedding_dim=256, in_channels=512, dataset_name='pascal_voc', \
                        num_classes=21, num_atrr_clusters=112, json_file='./gpt4.0_cluster_a_photo_of4.json',\
                        img_size=320, mode='train', device='cuda', cap_dir=None, fuse_weight=0.2, aug_first=False,
                        fuse_ver=1, extract_noun=False, gamma=0.3, refine_cam=False, w_noise=0):

        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)
        self.encoder.visual.reload_self_attn(layers=6, feat_size=img_size//16, mode=mode)
        self.encoder.eval()
        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=12)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        text_prompts = new_class_names+BACKGROUND_CATEGORY if num_classes <= 21 else new_class_names_coco+BACKGROUND_CATEGORY_COCO
        self.integral_text_features = clip.encode_text_with_prompt_ensemble(self.encoder, text_prompts, device, prompt_templates=['a clean origami {}.'])

        self.cap_dir = cap_dir
        self.fuse_weight = fuse_weight
        self.fuse_ver = fuse_ver
        self.extract_noun = extract_noun
        self.refine_cam = refine_cam
        self.gamma = gamma
        self.nlp = spacy.load("en_core_web_sm")
        if aug_first:
            self.text_attr, self.attr_flag = attr_aggregate(self.integral_text_features, dataset_name, num_classes-1, num_atrr_clusters, json_file)
        else:
            self.text_attr = self.integral_text_features.T

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups

    def refine_text(self, cls_labels, img_names):
        """
        Refines text with most similar caption

        Args:
            cls_labels (bs, C, ) : cls label for each img
            img_names (bs, ) : img name for each img

        Outputs:
            refined_text_feats (bs, C, D) : image specific text feat using captioner
        """
        all_text_feat_list = []
        for cls_label, img_name in zip(cls_labels, img_names):
            caption_dir = os.path.join(self.cap_dir, f"{img_name}.pickle")
            with open(caption_dir, "rb") as fr:
                specific_captions = pickle.load(fr)

            text_feat_list = []
            for i in range(self.text_attr.shape[-1]):
                if i >= 20 or cls_label[i] == 0:
                    text_feat_list.append(self.text_attr[:, i])
                    continue
                ref_cap = specific_captions[i][0]

                if self.extract_noun:
                    ref_cap = extract_noun_phrase(ref_cap, self.nlp, class_names[i])

                ref_cap_feat = zeroshot_classifier([ref_cap], ['a clean origami {}.'], self.encoder).squeeze(0)

                refined_feat = self.fuse_weight * ref_cap_feat + (1 - self.fuse_weight) * self.text_attr[:, i].cuda()
                text_feat_list.append(refined_feat)
            text_feats = torch.stack(text_feat_list, dim=0)
            all_text_feat_list.append(text_feats)
        
        all_text_feats = torch.stack(all_text_feat_list, dim=0)

        return all_text_feats

    def refine_text_ver2(self, caption_dir, cls_label):
        """
        Refines text with most similar caption

        Args:
            caption_dir : directory where captions are stored
            cls_labels (C, ) : cls label for each img

        Outputs:
            refined_text_feats (C, D) : image specific text feat using captioner
        """
        with open(caption_dir, "rb") as fr:
            specific_captions = pickle.load(fr)

        fg_text_feat_list = []
        for i in range(len(self.fg_text_features)):
            if cls_label[i] == 0:
                fg_text_feat_list.append(self.fg_text_features[i].cuda())
                continue
            ref_cap = specific_captions[i][0]

            if self.extract_noun:
                ref_cap = extract_noun_phrase(ref_cap, self.nlp, class_names[i])

            ref_cap_feat = zeroshot_classifier(ref_cap, ['a clean origami {}.'], self.encoder)
            sim = torch.mm(ref_cap_feat, self.fg_text_features[i].unsqueeze(-1))
            tgt_cap_feat = ref_cap_feat[sim.argmax(dim=0).item()]
            refined_feat = self.fuse_weight * tgt_cap_feat + (1 - self.fuse_weight) * self.fg_text_features[i].cuda()
            fg_text_feat_list.append(refined_feat)
        fg_text_feats = torch.stack(fg_text_feat_list, dim=0)
        return fg_text_feats


    def forward(self, img, ex_feats=None, img_names=None, cls_labels=None):
        if cls_labels is not None:
            text_attr = self.refine_text(cls_labels, img_names)
        else:
            text_attr = self.text_attr.T
        if ex_feats is not None:
            image_features_, attn_weights_, all_feats_ = clip.generate_clip_fts(img, self.encoder, return_weights=True, ex_feats=ex_feats)
            attr_maps_raw_ = clip.clip_feature_surgery(image_features_, text_attr)[:,1:,:self.num_classes-1]
            return attr_maps_raw_

        b, c, h, w = img.shape
        self.encoder.eval()
        image_features, attn_weights, all_feats = clip.generate_clip_fts(img, self.encoder, return_weights=True)

        if self.refine_cam:
            attr_maps_raw_cap = clip.clip_feature_surgery(image_features, text_attr)[:,1:,:self.num_classes-1]
            attr_maps_raw_org = clip.clip_feature_surgery(image_features, self.text_attr.T)[:,1:,:self.num_classes-1]


            attr_maps_raw = torch.where(
                attr_maps_raw_cap > self.gamma,
                (attr_maps_raw_org + attr_maps_raw_cap) / 2,
                attr_maps_raw_org
            )
        else:
            attr_maps_raw = clip.clip_feature_surgery(image_features, text_attr)[:,1:,:self.num_classes-1]

        all_img_tokens =  all_feats[:, :, 1:, ...]
        all_img_tokens = all_img_tokens.permute(0, 1, 3, 2)
        all_img_tokens = all_img_tokens.reshape(12, b, all_img_tokens.size(-2), h//16, w //16) #(11, b, c, h, w)

        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        seg, seg_attn_weight_list = self.decoder(fts)
        
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_fts_flatten = F.normalize(attn_fts_flatten, dim=1)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = (attn_pred - torch.mean(attn_pred) * 1.) * 3.0
        attn_pred = torch.sigmoid(attn_pred)

        return seg, attn_fts.clone().detach(), attr_maps_raw, attn_weights, attn_pred