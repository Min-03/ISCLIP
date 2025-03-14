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

from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from models.model import ISCLIP
import warnings


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/your/path/WeCLIP/configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument("--ft_layers", default=2, type=int, help="number of layers in fusion transformer")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers for dataloader")
parser.add_argument("--m_weight", default=0.1, type=float, help="loss weight for matching loss")
parser.add_argument("--match_ratio", default=0.75, type=float, help="match raitio used for parsed caption matching")
parser.add_argument("--fuse_ver", default=1, type=int, help="which model to use for refining prompts")
parser.add_argument("--cap_ver", default=1, type=int)
parser.add_argument("--fuse_mode", default="txt", type=str, help="which option(txt, cls_txt, img...) to use for refining prompts")
parser.add_argument("--refine_always", action="store_true", help="whether to refine CLIP visual encoder's attention until end")
parser.add_argument("--refine_bg", action="store_true", help="whether to refine background prompts with caption")
parser.add_argument("--refine_all", action="store_true", help="whether to refine other non-gt forground prompts with caption")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--use_raw", action="store_true")

warnings.filterwarnings(action='ignore', category=UserWarning)
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None, test_scales=[1, 0.75]):

    _preds, _gts, _msc_preds, cams = [], [], [], []
    
    model.cuda(0)
    model.eval()

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    for idx, data in enumerate(data_loader):
        num+=1

        name, inputs, labels, cls_labels = data
        names = name+name

        inputs = inputs.cuda()
        labels = labels.cuda()

        #######
        # resize long side to 512
        
        _, _, h, w = inputs.shape
        ratio = args.resize_long / max(h,w)
        _h, _w = int(h*ratio), int(w*ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)
        
        #######

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_cat, cam, attn_loss = model(inputs_cat, names, mode = 'val')
        
        cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, h, w = segs_cat.shape
        _, h_c, w_c = cam.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_cat, cam_cat, attn_loss = model(inputs_cat, names, mode='val')

                _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                segs_list.append(_segs)
                

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)
        

        cams += list(cam.cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))


        if num % 100 == 0:
            # _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
            # _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
            # _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
            # _preds, _gts, _msc_preds, cams = [], [], [], []
            print(f"Done {num} out of {len(data_loader)}", flush=True)
        
    _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
    _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
    _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
    model.train()
            
    return seg_score, msc_seg_score, cam_score
                

def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


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
    return F.cross_entropy(logits, gt)



def train(cfg):

    num_workers = args.num_workers
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    if args.debug:
        cfg.train.max_iters = 30000
    
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
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
    max_refine_iter = cfg.train.max_iters if args.refine_always else 15000
    ISCLIP_model = ISCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda',
        n_layers=args.ft_layers,
        match_ratio=args.match_ratio,
        fuse_ver=args.fuse_ver,
        fuse_mode=args.fuse_mode,
        max_refine_iter=max_refine_iter,
        refine_bg=args.refine_bg,
        refine_all=args.refine_all,
        use_raw=args.use_raw,
        cap_ver=args.cap_ver
    )
    
    # logging.info('\nNetwork config: \n%s'%(WeCLIP_model))
    param_groups = ISCLIP_model.get_param_groups()
    ISCLIP_model.cuda()


    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[4],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            }
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    # logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()
    


    for n_iter in range(cfg.train.max_iters):
        
        try:
            img_name, inputs, cls_labels, img_box, captions = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box, captions = next(train_loader_iter)

        mode = "train" if not args.debug else "debug"
        segs, cam, attn_pred= ISCLIP_model(inputs.cuda(), img_name, captions, mode=mode, cls_labels=cls_labels)

        pseudo_label = cam

        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        fts_cam = cam.clone()

            
        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        

        loss = 1 * seg_loss + 0.1*attn_loss


        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size


            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'), seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item()}, global_step=n_iter)

        
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "ISCLIP_model_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            torch.save(ISCLIP_model.state_dict(), ckpt_name)
            if (n_iter + 1) > 20000:
                torch.save(ISCLIP_model.state_dict(), ckpt_name)
            seg_score, msc_seg_score, cam_score = validate(model=ISCLIP_model, data_loader=val_loader, cfg=cfg)
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)
            logging.info("msc segs score")
            logging.info(msc_seg_score)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    # cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    # cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    # cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)
    
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.ckpt_dir)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.tb_logger_dir)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp, 'train.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)
