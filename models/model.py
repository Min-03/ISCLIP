import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import clip
from .fuse_transformer import TextFusionTransformer_ver1, TextFusionTransformer_ver2, TextFusionTransformer_ver3, TextFusionTransformer_ver4, TextFusionTransformer_ver5
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY, new_class_names_coco, BACKGROUND_CATEGORY_COCO
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam, perform_single_coco_cam
import os
import sys
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from models.PAR import PAR
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
import pickle




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
    

class ISCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda', 
                 n_layers=2, match_ratio=0.75, fuse_ver=1, fuse_mode="txt", max_refine_iter=15000, dataset="VOC", refine_bg=False):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)
        self.encoder = self.encoder.to(torch.float32)

        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

        self.in_channels = in_channels
        
        self.dataset = dataset

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        fuse_trans_dict = {1:TextFusionTransformer_ver1, 2:TextFusionTransformer_ver2, 3:TextFusionTransformer_ver3, 4:TextFusionTransformer_ver4, 5:TextFusionTransformer_ver5}
        self.fuse_transformer_fg = fuse_trans_dict.get(fuse_ver)(embed_dim=512, heads=8, layers=n_layers)
        self.fuse_transformer_bg = fuse_trans_dict.get(fuse_ver)(embed_dim=512, heads=8, layers=n_layers)

        bg_names = BACKGROUND_CATEGORY if dataset == "VOC" else BACKGROUND_CATEGORY_COCO
        fg_names = new_class_names if dataset == "VOC" else new_class_names_coco
        self.bg_text_features = zeroshot_classifier(bg_names, ['a clean origami {}.'], self.encoder)
        self.fg_text_features = zeroshot_classifier(fg_names, ['a clean origami {}.'], self.encoder)

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClassAug')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.cam_func_dict = {"VOC":perform_single_voc_cam, "COCO":perform_single_coco_cam}
        self.iter_num = 0
        self.require_all_fts = True
        
        grammar = "ADJ_NOUN: {<DT>?<JJ>*<NN|NNS>+}"
        self.chunk_parser = RegexpParser(grammar)
        self.match_ratio = match_ratio
        assert fuse_mode in ["txt", "cls_txt", "txt_parsed", "img"], f"{fuse_mode} for refine text is not supported"
        self.fuse_mode = fuse_mode
        self.max_refine_iter = max_refine_iter
        if dataset != "VOC" and max_refine_iter == 15000:
            raise Exception("Something is wrong with your setting")
        self.caption_file_dir = "/data/dataset/VOC2012/SpecificCaption" if dataset == "VOC" else "/data/dataset/COCO_seg/SpecificCaption"
        self.refine_bg = refine_bg


    def get_param_groups(self):

        param_groups = [[], [], [], [], []]  # backbone; backbone_norm; cls_head; seg_head; fuse_transformer

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        for param in list(self.fuse_transformer_fg.parameters()):
            param_groups[4].append(param)
        for param in list(self.fuse_transformer_bg.parameters()):
            param_groups[4].append(param)
            
        return param_groups
    
    def refine_text(self, caption):
        """
        input
        caption : single caption string
        
        output :
        fg_text_features : (fg_num, clip_dim)
        bg_text_features : (bg_num, clip_dim)
        """
        caption_feat = zeroshot_classifier([caption], [''], self.encoder)
        caption_feat = caption_feat.cuda().to(torch.float32).detach()
        
        fg_text_features = self.fuse_transformer_fg(self.fg_text_features.cuda().detach().to(torch.float32), caption_feat, caption_feat)
        bg_text_features = self.fuse_transformer_bg(self.bg_text_features.cuda().detach().to(torch.float32), caption_feat, caption_feat)
        return fg_text_features, bg_text_features
    
    def refine_text_parsed(self, caption):
        """
        input
        caption : single caption string
        
        output
        fg_text_features : (fg_num, clip_dim)
        bg_text_features : (bg_num, clip_dim)
        """
        caption_feat = zeroshot_classifier([caption], [''], self.encoder)
        caption_feat = caption_feat.cuda().to(torch.float32).detach()
        
        words = word_tokenize(caption)
        tagged_words = pos_tag(words)
        tree = self.chunk_parser.parse(tagged_words)
        phrases = [" ".join(word for word, tag in subtree.leaves()) for subtree in tree.subtrees() if subtree.label() == "ADJ_NOUN"]
        
        if len(phrases) == 0:
            fg_text_features = self.fuse_transformer_fg(self.fg_text_features.cuda().detach().to(torch.float32), caption_feat, caption_feat)
            bg_text_features = self.fuse_transformer_bg(self.bg_text_features.cuda().detach().to(torch.float32), caption_feat, caption_feat)
            return fg_text_features, bg_text_features
            
        parsed_caption_feat = zeroshot_classifier(phrases, ['a clean origami {}.'], self.encoder).detach().cuda().to(torch.float32)
        sim = torch.matmul(parsed_caption_feat, self.fg_text_features.T).detach() #(pnum, F)
        match_sim, match_idxs = torch.max(sim, dim=1)
        
        fg_num = self.fg_text_features.shape[0]
        refine_idx = [[] for _ in range(fg_num)]
        fg_text_feat_list = []
        for i in range(len(phrases)):
            if match_sim[i] < self.match_ratio:
                continue
            refine_idx[match_idxs[i]].append(i)
            
        for i in range(fg_num):
            if len(refine_idx[i]) == 0:
                fg_text_feat_list.append(self.fg_text_features[i].cuda().detach().to(torch.float32))
            else:
                ref_cap_list = [parsed_caption_feat[idx] for idx in refine_idx[i]]
                ref_caps = torch.stack(ref_cap_list, dim=0)
                refined_feat = self.fuse_transformer_fg(self.fg_text_features[i].cuda().detach().to(torch.float32).unsqueeze(0), ref_caps, ref_caps)
                fg_text_feat_list.append(refined_feat.squeeze(0))
        fg_text_features = torch.stack(fg_text_feat_list, dim=0)
        bg_text_features = self.fuse_transformer_bg(self.bg_text_features.cuda().detach().to(torch.float32), caption_feat, caption_feat)
        return fg_text_features, bg_text_features
    
    def refine_text_with_img(self, cam_fts):
        """
        input
        img_feat : single cam feature (hw, 1, clip_dim)
        
        output
        fg_text_features : (fg_num, clip_dim)
        bg_text_features : (bg_num, clip_dim)
        """
        self.grad_cam.activations_and_grads.release()
        with torch.no_grad():
            logits, _, img_feature = self.encoder.forward_last_layer(cam_fts, self.fg_text_features, require_img=True)
            img_feature = img_feature.to(torch.float32)
        self.grad_cam.activations_and_grads.register(self.target_layers)
        fg_text_features = self.fuse_transformer_fg(self.fg_text_features.cuda().detach().to(torch.float32), img_feature, img_feature)
        bg_text_features = self.fuse_transformer_bg(self.bg_text_features.cuda().detach().to(torch.float32), img_feature, img_feature)
        return fg_text_features, bg_text_features
    
    def refine_text_specific(self, caption, caption_dir, cls_label):
        """
        Args:
            caption : single caption string
            caption_dir : directory path where class specific captions stored
            cls_label (C,) : one hot class label
        """
        caption_feat = zeroshot_classifier([caption], [''], self.encoder)
        caption_feat = caption_feat.cuda().to(torch.float32).detach()
        
        with open(caption_dir, "rb") as fr:
            specific_captions = pickle.load(fr)
        specific_caption_feats = zeroshot_classifier(list(specific_captions.values()), [''], self.encoder)
        specific_caption_feats = specific_caption_feats.cuda().to(torch.float32).detach()
        
        fg_text_feat_list = []
        cap_num = 0
        for i in range(len(self.fg_text_features)):
            if cls_label[i] == 0:
                fg_text_feat_list.append(self.fg_text_features[i].cuda().detach().to(torch.float32))
                continue
            ref_cap = specific_caption_feats[cap_num].unsqueeze(0)
            cap_num += 1
            refined_feat = self.fuse_transformer_fg(self.fg_text_features[i].cuda().detach().to(torch.float32).unsqueeze(0), ref_cap, ref_cap, class_idx=i)
            fg_text_feat_list.append(refined_feat.squeeze(0))
        fg_text_features = torch.stack(fg_text_feat_list, dim=0)
        
        bg_text_features = self.bg_text_features.cuda().detach().to(torch.float32)
        if self.refine_bg:
            bg_text_features = self.fuse_transformer_bg(bg_text_features, caption_feat, caption_feat)
        return fg_text_features, bg_text_features

    
    def forward(self, img, img_names='2007_000032', captions=None, mode='train', requires_prompt=False, cls_labels=None):
        cam_list = []
        prompts_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1
        
        if self.iter_num % 2000 == 0:
            print(self.fg_text_features)

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(b, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)


        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        
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
            
            if mode == 'train':
                if self.fuse_mode=="img":
                    fg_text_feats, bg_text_feats = self.refine_text_with_img(cam_fts)
                elif self.fuse_mode=="txt":
                    caption = captions[i]
                    fg_text_feats, bg_text_feats = self.refine_text(caption)
                elif self.fuse_mode=="cls_txt":
                    caption = captions[i]
                    cls_label = cls_labels[i]
                    caption_dir = os.path.join(self.caption_file_dir, f"{img_name}.pickle")
                    fg_text_feats, bg_text_feats = self.refine_text_specific(caption, caption_dir, cls_label)
                else:
                    caption = captions[i]
                    fg_text_feats, bg_text_feats = self.refine_text_parsed(caption)
            else:
                fg_text_feats, bg_text_feats = self.fg_text_features, self.bg_text_features
            
            if self.iter_num > self.max_refine_iter or mode=='val': #15000
                require_seg_trans = True
            else:
                require_seg_trans = False

            cam_refined_list, keys, w, h = self.cam_func_dict.get(self.dataset)(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   bg_text_feats, fg_text_feats,
                                                                   self.grad_cam,
                                                                   mode=mode,
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
            
            
            refined_prompts = torch.cat((fg_text_feats, bg_text_feats), dim=0)
            prompts_list.append(refined_prompts)
            
        all_cam_labels = torch.stack(cam_list, dim=0)

        if not requires_prompt:
            return seg, all_cam_labels, attn_pred
        
        all_refined_prompts = torch.stack(prompts_list, dim=0)
        prompts_org = torch.cat((self.fg_text_features, self.bg_text_features), dim=0).to(torch.float32).detach()
        prompts = [prompts_org, all_refined_prompts]
        return seg, all_cam_labels, attn_pred, prompts

        
