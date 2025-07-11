import os

import sys
import time
import argparse
from config import get_config
from models import build_model
import torch
import torch.nn as nn
from torchvision import transforms as trans

from utilss import evaluate, get_dataset, FFDataset, setup_logger
import numpy as np
import random
import warnings


from modules.preprocess import process_image

from modules.blockSA import MultiScaleEEM
from modules.inter_att import CrossModalAttention
from modules.attention import ChannelwiseCosineSimilarityScoring


warnings.filterwarnings("ignore")

num_gpus = torch.cuda.device_count()
gpu_ids = [*range(num_gpus)]

pretrained_path = "/home/admin1/DFMamba/pretrained/vssm_small_0229_ckpt_epoch_222.pth"


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

args, config = parse_option()

import torch
import random


class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        # 初始化参数
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("比率必须是浮点数或两个浮点数的元组。")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor1, tensor2):
        # 确保输入是两个张量
        if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
            raise ValueError("输入必须是两个PyTorch张量。")

        # 检查输入形状是否匹配 (B, C, H, W)
        if tensor1.shape != tensor2.shape or len(tensor1.shape) != 4:
            raise ValueError("两个输入张量必须具有相同的BCHW形状。")

        # 如果随机概率未触发，直接返回原张量
        if random.random() > self.p:
            return tensor1, tensor2

        b, c, h, w = tensor1.shape

        # 创建输出张量的副本
        masked_tensor1 = tensor1.clone()
        masked_tensor2 = tensor2.clone()

        # 为batch中的每个样本单独生成掩膜
        for batch_idx in range(b):
            # 确定掩膜比率
            if self.fixed_ratio:
                ratio = self.ratio[0]
            else:
                ratio = random.uniform(self.ratio[0], self.ratio[1])

            # 计算需要的掩膜总数
            num_masks = int((h * w * ratio) / (self.patch_size ** 2))

            # 生成不重叠的随机位置集合
            all_positions = set()
            while len(all_positions) < num_masks:
                top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
                left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
                all_positions.add((top, left))

            # 将位置随机分配给两个掩膜，确保互不重叠
            positions_list = list(all_positions)
            random.shuffle(positions_list)
            split_point = len(positions_list) // 2
            positions1 = positions_list[:split_point]
            positions2 = positions_list[split_point:]

            # 对第一个张量应用掩膜
            for (top, left) in positions1:
                masked_tensor1[batch_idx, :, top:top + self.patch_size, left:left + self.patch_size] = 0

            # 对第二个张量应用掩膜
            for (top, left) in positions2:
                masked_tensor2[batch_idx, :, top:top + self.patch_size, left:left + self.patch_size] = 0

        return masked_tensor1, masked_tensor2

class Dual_network(nn.Module):
    def __init__(self, is_training):
        super().__init__()
        
        self.is_training = is_training
                
        #######################################################################################################
                
        #self.mask = RandomMask(ratio=0.25, patch_size=8, p=0.5)

        self.sobel1 = MultiScaleEEM(192)
        self.sobel2 = MultiScaleEEM(192)

        self.inter1 = CrossModalAttention(192, height=28, width=28)
        self.inter2 = CrossModalAttention(384, height=14, width=14)
        self.inter3 = CrossModalAttention(768, height=7, width=7)
        
        
        self.channel_enhance = ChannelwiseCosineSimilarityScoring(in_planes=768, ratio=8, temperature=1)
        


    def init_model_rgb(self):
        self.model_rgb = build_model(config)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        try:

            current_model_dict = self.model_rgb.state_dict()
            pretrained_weights = torch.load(pretrained_path, map_location=torch.device("cpu"))
            #print(f"Successfully load ckpt")

            pretrained_weights['model'].pop('classifier.head.weight')
            pretrained_weights['model'].pop('classifier.head.bias')
            # print(pretrained_weights)

            self.model_rgb.load_state_dict(pretrained_weights['model'], strict=False)
            # print(model.state_dict()['layers.2.blocks.9.mlp.fc1.weight'])
            print('Successfully loading！', end=' ')
        except Exception as e:
            print(f"Failed loading checkpoint: {e}")
            
            

    def forward(self, x):   
            
        y = process_image(x)
        
        
        #if self.is_training:
            #x, y = self.mask(x, y)

        ###########################################
        x1 = self.model_rgb.part_1(x)
        
        y1 = self.model_rgb.part_1(y)
        
        x1 = self.sobel1(x1)
        
        y1 = self.sobel2(y1)
        
        x1, y1 = self.inter1(x1, y1)

        
        ###############################################
        x2 = self.model_rgb.part_2(x1)
        
        y2 = self.model_rgb.part_2(y1) 
        
        x2, y2 = self.inter2(x2, y2) 

        ###############################################
        x3 = self.model_rgb.part_3(x2)
        
        y3 = self.model_rgb.part_3(y2)
        
        x3, y3 = self.inter3(x3, y3) 

        ###############################################
        x4 = self.model_rgb.part_4(x3)
        
        y4 = self.model_rgb.part_4(y3)
        
        x = self.channel_enhance(x4, y4)
        
        #x = x4 + y4
        
        
        #x = self.aff_fusion1(x4, y4)
        
        
        #x = self.fusion1(x4, y4)

        
        ###############################################
        x = self.model_rgb.classifier_out1(x)
        
   
        return x


if __name__ == '__main__':
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    batch_size, channels, height, width = 32, 3, 224, 224
    x = torch.rand(batch_size, channels, height, width)
    x = x.to(device)
    dual_net = Dual_network()
    dual_net.init_model_rgb()
    dual_net = dual_net.to(device)
    a = dual_net(x)
    print(a.shape)


