import os

import sys
import time
import argparse
from config import get_config
from models import build_model
import torch, gc
import torch.nn as nn
import torch.nn.functional as F

from utilss import evaluate, get_dataset, FFDataset, setup_logger, evaluate1, evaluate2, evaluate3, evaluate4, evaluate5, evaluate6, evaluate7
import numpy as np
import random
import warnings
from dual import Dual_network
from torch.utils.data import ConcatDataset, DataLoader, Subset

gc.collect()
torch.cuda.empty_cache()

warnings.filterwarnings("ignore")

# 配置
dataset_path = '/home/admin1/DFMamba/dataset/FF++c23/'

dataset_path1 = "/home/admin1/DFMamba/dataset/FF++c40/"

dataset_path2 = "/home/admin1/DFMamba/dataset/CelebDF-v2/"

dataset_path3 = "/home/admin1/DFMamba/dataset/WildDeepfake/"

dataset_path4 = "/home/admin1/DFMamba/dataset/DFDCp/"

batch_size = 32
num_gpus = torch.cuda.device_count()
gpu_ids = [*range(num_gpus)]
max_epoch = 30
loss_freq = 40
ckpt_dir = '/home/admin1/DFMamba/logs/FF++c23/'
ckpt_name = 'VMamba/nt'

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def initModel(mod, gpu_ids):
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        default="/home/admin1/DFMamba/configs/vssm/vmambav2_small_224.yaml",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters 指数滑动平均
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


if __name__ == '__main__':
    args, config = parse_option()
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    model = Dual_network(is_training=True)
    model.init_model_rgb()
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

    # dataset
    dataset_real = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'), size=224, frame_num=32, augment=True)
    dataset_fake, _ = get_dataset(name='train', size=224, root=dataset_path, frame_num=32, augment=True)
    
    # label
    labels_real = torch.zeros(len(dataset_real))  # 真实数据标签为0
    labels_fake = torch.ones(len(dataset_fake))   # 伪造数据标签为1

    # combine
    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_real, dataset_fake, labels_real, labels_fake):
            self.data = ConcatDataset([dataset_real, dataset_fake])
            self.labels = torch.cat([labels_real, labels_fake], dim=0)
            # 创建索引并打乱
            self.indices = list(range(len(self.data)))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # 使用打乱后的索引
            mapped_idx = self.indices[idx]
            return self.data[mapped_idx], self.labels[mapped_idx]

    combined_dataset = CombinedDataset(dataset_real, dataset_fake, labels_real, labels_fake)

    # dataloader
    dataloader = DataLoader(
        dataset=combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    len_dataloader = len(dataloader)
    print(len_dataloader)

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'train.log', 'logger')
    best_val = 0.

    # train
    model.total_steps = 0
    epoch = 0

    while epoch < max_epoch:
        model.train()
        logger.debug(f'No {epoch + 1}')
        i = 0

        for data, label in dataloader:
            i += 1
            model.total_steps += 1

            data = data.detach()
            label = label.detach()
            

            data = data.to(device)
            label = label.to(device)

            stu_cla = model(data)
            loss_cla = loss_fn(stu_cla.squeeze(1), label)  # 分类损失

            loss = loss_cla

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')

        # eval
        model.eval()
        #model.is_training = False
        
        auc, r_acc, f_acc, t_acc = evaluate(model, dataset_path, mode='test')
        logger.debug(f'(Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')
        
        #auc, r_acc, f_acc, t_acc = evaluate1(model, dataset_path, mode='test')
        #logger.debug(f'(Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')

        #auc, r_acc, f_acc, t_acc = evaluate2(model, dataset_path, mode='test')
        #logger.debug(f'(Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')
        
       # auc, r_acc, f_acc, t_acc = evaluate3(model, dataset_path, mode='test')
        #logger.debug(f'(Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')
        
        
        
        
        #auc, r_acc, f_acc, t_acc = evaluate4(model, dataset_path1, mode='test')
        #logger.debug(f'(FF++c40 Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')
       
        #auc, r_acc, f_acc, t_acc = evaluate5(model, dataset_path2, mode='test')
        #logger.debug(f'(Celebv2 Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')
        
        #auc, r_acc, f_acc, t_acc = evaluate6(model, dataset_path3, mode='test')
        #logger.debug(f'(WildDeepfake Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')
        
        #auc, r_acc, f_acc, t_acc = evaluate7(model, dataset_path4, mode='test')
        #logger.debug(f'(DFDCp Test @ epoch {epoch + 1}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, t_acc:{t_acc}')

        #model.is_training = True

        # save model
        ckpt_model_name = f'train_epoch_{epoch + 1}.pkl'
        torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_model_name))


        epoch += 1