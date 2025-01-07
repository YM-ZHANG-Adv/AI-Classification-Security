#加载模型并获取dataloader
from tools.import_model import trmodel
from attack.CW import CWAttack
import numpy as np # type: ignore
import cv2# type: ignore
import torchattacks # type: ignore
import argparse
import pathlib
import time
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import PIL.Image # type: ignore
import torch.nn.functional as F # type: ignore
import sys
sys.argv = ['run.py']

import numpy as np # type: ignore
import torch# type: ignore
import torch.nn as nn # type: ignore
import torch.distributed as dist# type: ignore
import torchvision# type: ignore
import cv2# type: ignore

from fvcore.common.checkpoint import Checkpointer# type: ignore

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)
from pytorch_image_classification.config.config_node import ConfigNode
from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    compute_accuracy,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)
from pytorch_image_classification import (
    get_default_config,
    create_model,
    create_transform,
)
import torch.nn.functional as F
import copy
import os

os.chdir('/root/ZYM/zym/AI_S/torch_classification')
cfg_path = './configs/self_dataset/resnet.yaml'
pth_path = './experiments/mnist/resnet/exp06/checkpoint_00160.pth'

#水印嵌入模型仍为resnet110
model = trmodel(cfg_path,pth_path).get_model().to(torch.device("cuda:0"))
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default = '/root/ZYM/zym/AI_S/torch_classification/configs/self_dataset/resnet.yaml' )
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    config.freeze()
    return config

config = load_config()
set_seed(config)
setup_cudnn(config)
from pytorch_image_classification.datasets.datasets_custom import create_dataset
_dataset = create_dataset(config, is_train = True)[0] #水印训练数据集，从训练集中挑选总数据集的1%，即600张
from torch.utils.data import Subset, DataLoader, random_split
num_samples = int(len(_dataset) * 0.01) 
indices = torch.randperm(len(_dataset))[:num_samples]
wm_dataset = Subset(_dataset, indices)
wm_dataloader = DataLoader(wm_dataset, batch_size=32, shuffle=True)
class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_name):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.feature = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.feature = output.detach()
        
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
    def forward(self, x):
        self.model.eval()
        _ = self.model(x)
        return self.feature

model.train()
feature_extractor = FeatureExtractor(model, layer_name='stage1.block3')
fs =  feature_extractor(wm_dataset[12][0].unsqueeze(0).to(torch.device("cuda:0"))).shape[1:] #在此处额外获取feature的形状,此处为元组type
from watermark import DeepSignsWatermark
target_layer = 'stage1.block3'
deepsigns = DeepSignsWatermark(model,wm_dataloader,target_layer)
watermarked_model = deepsigns.wm_train(feature_shape= fs)