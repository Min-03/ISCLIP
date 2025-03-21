import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
from .fuse_transformer import TextFusionTransformer, TextFusionTransformer_ver2, TextFusionTransformer_ver3
import numpy as np
import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_model.PAR import PAR
from sklearn.mixture import GaussianMixture
import pickle
from utils.camutils import draw_activation
import matplotlib.pyplot as plt
from utils.imutils import denormalize_img



def minmax_norm(inputs):
    return (inputs - inputs.min()) / (inputs.max() - inputs.min())

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
                 fuse_ver=1, fuse_mode="txt", caption_file_dir=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

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
        
        ##################### added #####################
        fuse_trans_dict = {1:TextFusionTransformer, 2:TextFusionTransformer_ver2, 3:TextFusionTransformer_ver3}
        self.fuse_transformer = fuse_trans_dict.get(fuse_ver)(embed_dim=512, heads=8, layers=2).to(torch.float16)
        self.caption_file_dir = caption_file_dir
        self.gamma = 0.80
        self.fuse_mode = fuse_mode

    def get_param_groups(self):

        param_groups = [[], [], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
            
        for param in list(self.fuse_transformer.parameters()):
            param_groups[4].append(param)

        return param_groups
    
    def refine_text_with_img(self, cam_fts, cls_label):
        """
        input
        img_feat : single cam feature (hw, 1, clip_dim)
        
        output
        fg_text_features : (fg_num, clip_dim)
        bg_text_features : (bg_num, clip_dim)
        """
        self.grad_cam.activations_and_grads.release()
        with torch.no_grad():
            img_feature = self.encoder.forward_img_last_layer(cam_fts).squeeze(0)
            similarities_fg = self.encoder.get_logits(img_feature, self.fg_text_features)
        self.grad_cam.activations_and_grads.register(self.target_layers)
    
        fg_text_feat_list = []
        for i in range(len(self.fg_text_features)):
            if cls_label[i] == 0:
                fg_text_feat_list.append(self.fg_text_features[i].cuda())
                continue
            ref_img_feat_fg = self.get_filtered_img_feat(img_feature, similarities_fg[:, i].unsqueeze(-1))
            refined_feat = self.fuse_transformer(self.fg_text_features[i].cuda().unsqueeze(0), ref_img_feat_fg, ref_img_feat_fg, class_idx=i)
            fg_text_feat_list.append(refined_feat.squeeze(0))
        fg_text_features = torch.stack(fg_text_feat_list, dim=0)
        
        bg_text_features = self.bg_text_features.cuda()
        
        return fg_text_features, bg_text_features
    
    def refine_text_and_vis(self, caption_dir, cls_label, cam_fts, img):
        
        with open(caption_dir, "rb") as fr:
            specific_captions = pickle.load(fr)
        specific_caption_feats = zeroshot_classifier(list(specific_captions.values()), ['{}'], self.encoder)
        fg_text_feat_list = []
        cap_num = 0
        
        self.grad_cam.activations_and_grads.release()
        with torch.no_grad():
            img_feature = self.encoder.forward_img_last_layer(cam_fts).squeeze(0)
        self.grad_cam.activations_and_grads.register(self.target_layers)
        
        for i in range(len(self.fg_text_features)):
            if cls_label[i] == 0:
                fg_text_feat_list.append(self.fg_text_features[i].cuda())
                continue
            ref_cap = specific_caption_feats[cap_num].unsqueeze(0)
            cap_num += 1
            refined_feat = self.fuse_transformer(self.fg_text_features[i].cuda().unsqueeze(0), ref_cap, ref_cap, class_idx=i)
            fg_text_feat_list.append(refined_feat.squeeze(0))
            
            with torch.no_grad():
                similarities_fg = self.encoder.get_logits(img_feature, self.fg_text_features[i].unsqueeze(0)).squeeze(-1)
                n = int(similarities_fg.shape[0] ** 0.5)
                similarities_fg = similarities_fg.reshape(1, 1, n, n)
                similarities_fg = F.interpolate(similarities_fg, size=img.shape[1:], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            img_path = caption_dir.replace("SpecificCaption2", "cam_results").replace(".pickle", f"_{new_class_names[i]}.png")
            print("before")
            print(similarities_fg)
            print("after")
            print(minmax_norm(similarities_fg))
            draw_activation(img, minmax_norm(similarities_fg), img_path, list(specific_captions.values())[cap_num - 1])
        fg_text_features = torch.stack(fg_text_feat_list, dim=0)
        
        bg_text_features = self.bg_text_features.cuda()

        return fg_text_features, bg_text_features
    
    def refine_text(self, caption_dir, cls_label):
        """
        Args:
            caption : single caption string
            caption_dir : directory path where class specific captions stored
            cls_label (C,) : one hot class label
        """
        
        with open(caption_dir, "rb") as fr:
            specific_captions = pickle.load(fr)
        specific_caption_feats = zeroshot_classifier(list(specific_captions.values()), ['{}'], self.encoder)
        fg_text_feat_list = []
        cap_num = 0
        for i in range(len(self.fg_text_features)):
            if cls_label[i] == 0:
                fg_text_feat_list.append(self.fg_text_features[i].cuda())
                continue
            ref_cap = specific_caption_feats[cap_num].unsqueeze(0)
            cap_num += 1
            refined_feat = self.fuse_transformer(self.fg_text_features[i].cuda().unsqueeze(0), ref_cap, ref_cap, class_idx=i)
            fg_text_feat_list.append(refined_feat.squeeze(0))
        fg_text_features = torch.stack(fg_text_feat_list, dim=0)
        
        bg_text_features = self.bg_text_features.cuda()

        return fg_text_features, bg_text_features
    
    def get_filtered_img_feat(self, img_feat, sim):
        gmm = GaussianMixture(n_components=2, max_iter=100, random_state=1)
        gmm.fit(sim.cpu().detach().numpy())
        prob = gmm.predict_proba(sim.cpu().detach().numpy())
        high_sim = torch.tensor(prob[:, gmm.means_.argmax()] > self.gamma)
        ref_img_feat = img_feat[high_sim, :].mean(dim=0, keepdim=True)
        return ref_img_feat

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
                cls_label = cls_labels[i]
                if self.fuse_mode=="img":
                    fg_text_feats, bg_text_feats = self.refine_text_with_img(cam_fts, cls_label)
                else:
                    caption_dir = os.path.join(self.caption_file_dir, f"{img_name}.pickle")
                    # fg_text_feats, bg_text_feats = self.refine_text_and_vis(caption_dir, cls_label, cam_fts, img_i)
                    fg_text_feats, bg_text_feats = self.refine_text(caption_dir, cls_label)
            else:
                fg_text_feats, bg_text_feats = self.fg_text_features, self.bg_text_features
                cls_label = cls_labels[i]
                # cls_label = None

            cam_mode = "train" if mode == "debug" else mode
            cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   bg_text_feats, fg_text_feats,
                                                                   self.grad_cam,
                                                                   mode=cam_mode,
                                                                   require_seg_trans=require_seg_trans,
                                                                   cls_label=cls_label)


            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()
            
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            
            ##############
            if mode == "train":
                with open(caption_dir, "rb") as fr:
                    specific_captions = pickle.load(fr)
                
            img_denorm = denormalize_img(img[i].unsqueeze(0)).squeeze(0)
            img_denorm = img_denorm.permute(1, 2, 0).cpu().detach().numpy()
            plt.imshow(img_denorm)
            plt.savefig(os.path.join("/data/dataset/VOC2012/cam_results", str(img_name) + "_raw" + ".png"))
            plt.close()
            # print("cls", torch.nonzero(cls_labels[i]))
            cam_num = 1
            for idx in torch.nonzero(cls_labels[i]):
                # print(specific_captions.keys())
                # print(specific_captions)
                # print("idx:", idx, idx.shape)
                # for k in specific_captions.keys():
                #     print("keys:", k, k.shape)
                title = list(specific_captions.values())[cam_num - 1] if mode == "train" else new_class_names[idx]
                # print(cams.shape)
                if cam_num >= cams.shape[0]:
                    print(img_name, new_class_names[idx], "!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    continue
                draw_activation(img_i, cams[cam_num], os.path.join("/data/dataset/VOC2012/cam_results", str(img_name) + "_" + new_class_names[idx] + ".png"), title)
                cam_num += 1
            #############
                
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
                
            
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        return seg, all_cam_labels, attn_pred

        
