import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY, class_names
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_model.PAR import PAR
from utils.nlp_utils import extract_noun_phrase
import pickle
import spacy



def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



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


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class WeCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda',
                caption_dir=None, fuse_weight=0.1, cam_fuse_weight=0.5, fuse_ver=1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.encoder)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.encoder)

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClassAug')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True

        self.caption_file_dir = caption_dir
        self.fuse_weight = fuse_weight
        self.cam_fuse_weight = cam_fuse_weight
        self.fuse_ver = fuse_ver
        self.nlp = spacy.load("en_core_web_sm")

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups
    
    def refine_text(self, caption_dir, cls_label):
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
            ref_cap = specific_captions[i]
            ref_cap_feat = zeroshot_classifier(ref_cap, ['a clean origami {}.'], self.encoder)
            sim = torch.mm(ref_cap_feat, self.fg_text_features[i].unsqueeze(-1))
            tgt_cap_feat = ref_cap_feat[sim.argmax(dim=0).item()]
            refined_feat = self.fuse_weight * tgt_cap_feat + (1 - self.fuse_weight) * self.fg_text_features[i].cuda()
            fg_text_feat_list.append(refined_feat)
        fg_text_feats = torch.stack(fg_text_feat_list, dim=0)
        return fg_text_feats

    def refine_text_ver2(self, caption_dir, cls_label):
        """
        Refines text with target noun/detailed noun extracted from caption
        
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
            ref_cap = extract_noun_phrase(ref_cap, self.nlp, class_names[i])
            ref_cap_feat = zeroshot_classifier(ref_cap, ['a clean origami {}.'], self.encoder)
            tgt_cap_feat = ref_cap_feat[0]
            refined_feat = self.fuse_weight * tgt_cap_feat + (1 - self.fuse_weight) * self.fg_text_features[i].cuda()
            fg_text_feat_list.append(refined_feat)
        fg_text_feats = torch.stack(fg_text_feat_list, dim=0)
        return fg_text_feats

    def forward_with_fuse(self, img, img_names='2007_000032', mode='train', cls_labels=None):
        cam_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)


        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        seg, seg_attn_weight_list = self.decoder(fts)
        
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.png')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]
            
            if self.iter_num > 15000 or mode=='val': #15000
                require_seg_trans = True
            else:
                require_seg_trans = False
            
            cams_list = []
            for refine_mode in ["refine_fg", "None"]:
                if mode == "train" and refine_mode == "refine_fg":
                    caption_dir = os.path.join(self.caption_file_dir, f"{img_name}.pickle")
                    fg_text_feats = self.refine_text(caption_dir, cls_labels[i])
                else:
                    fg_text_feats = self.fg_text_features

                cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                    self.bg_text_features, fg_text_feats,
                                                                    self.grad_cam,
                                                                    mode=mode,
                                                                    require_seg_trans=require_seg_trans)


                cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
                
                cams = cam_dict['refined_cam'].cuda()
                cams_list.append(cams)

            cams_final = self.cam_fuse_weight * cams_list[0] + (1 - self.cam_fuse_weight) * cams_list[1] 
            bg_score = torch.pow(1 - torch.max(cams_final, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams_final = torch.cat([bg_score, cams_final], dim=0).cuda()
            
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams_final, valid_key)
            
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        return seg, all_cam_labels, attn_pred

    def forward(self, img, img_names='2007_000032', mode='train', cls_labels=None):
        cam_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)


        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        seg, seg_attn_weight_list = self.decoder(fts)
        
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.png')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]
            
            if self.iter_num > 15000 or mode=='val': #15000
                require_seg_trans = True
            else:
                require_seg_trans = False

            if mode == "train":
                caption_dir = os.path.join(self.caption_file_dir, f"{img_name}.pickle")
                if self.fuse_ver == 1:
                    fg_text_feats = self.refine_text(caption_dir, cls_labels[i])
                else:
                    fg_text_feats = self.refine_text_ver2(caption_dir, cls_labels[i])
            else:
                fg_text_feats = self.fg_text_features

            cam_mode = mode if mode != "no_cap" else "train"
            cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, fg_text_feats,
                                                                   self.grad_cam,
                                                                   mode=cam_mode,
                                                                   require_seg_trans=require_seg_trans)


            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()
            
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        return seg, all_cam_labels, attn_pred

        
