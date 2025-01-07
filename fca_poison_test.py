import os, sys
from os.path import abspath
import argparse
import pathlib
import time
from fvcore.common.checkpoint import Checkpointer # type: ignore
import torch

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



def load_config_notebook(): #不通过命令行参数的情况下获得config设置文件
    local_rank = 0
    config_path = '/root/ZYM/zym/AI_S/torch_classification/configs/self_dataset/resnet.yaml'
    config = get_default_config()
    if config_path is not None:
        config.merge_from_file(config_path)
    #config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    config.merge_from_list(['train.dist.local_rank', local_rank])
    config = update_config(config)
    config.freeze()
    return config

def subdivide_batch(config, data, targets):
    subdivision = config.train.subdivision

    if subdivision == 1:
        return [data], [targets]

    data_chunks = data.chunk(subdivision)
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        targets1, targets2, lam = targets
        target_chunks = [(chunk1, chunk2, lam) for chunk1, chunk2 in zip(
            targets1.chunk(subdivision), targets2.chunk(subdivision))]
    elif config.augmentation.use_ricap:
        target_list, weights = targets
        target_list_chunks = list(
            zip(*[target.chunk(subdivision) for target in target_list]))
        target_chunks = [(chunk, weights) for chunk in target_list_chunks]
    else:
        target_chunks = targets.chunk(subdivision)
    return data_chunks, target_chunks

def send_targets_to_device(config, targets, device):
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        t1, t2, lam = targets
        targets = (t1.to(device), t2.to(device), lam)
    elif config.augmentation.use_ricap:
        labels, weights = targets
        labels = [label.to(device) for label in labels]
        targets = (labels, weights)
    else:
        targets = targets.to(device)
    return targets


from pytorch_image_classification.datasets.datasets_custom import create_dataset
import numpy as np
import random
config = load_config_notebook()
poison_dataset = create_dataset(config, is_train = False) #将验证集作为投毒样本的来源
img_num = len(poison_dataset)
split_num = 1000 #将数据集按3：7的比例划分为中毒训练集和测试集

class_descr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#x为投毒训练集列表，y为测试集列表
train_list = []
train_label = []
test_list = []
test_label = []

dataset_length = len(poison_dataset)
for i in range(split_num):
    train_index = random.randint(0, dataset_length - 1)
    test_index = random.randint(0, dataset_length - 1)
    data_tensor,label_int = poison_dataset[train_index]
    test_data_tensor,test_label_int = poison_dataset[test_index]
    train_list.append(data_tensor)
    test_list.append(test_data_tensor)
    label_oh = np.zeros((10),dtype = np.float32) #创建one-hot标签向量
    label_oh[label_int] = np.float32(1)
    train_label.append(label_oh)
    label_oh2 = np.zeros((10),dtype = np.float32)
    label_oh2[test_label_int] = np.float32(1)
    test_label.append(label_oh2)

# 获取训练完成的分类模型resnet110
model = create_model(config)
model = apply_data_parallel_wrapper(config, model)
checkpoint = torch.load('/root/ZYM/zym/AI_S/torch_classification/experiments/mnist/resnet/exp06/checkpoint_00100.pth')
model.load_state_dict(checkpoint['model'], strict= False)
device = torch.device("cuda:0")
model.to(device)

# 设置基类为4，目标类为9，实现的效果为输入9输出4，用中毒的4训练
target_class = '4'
target_label = np.zeros(len(class_descr))
target_label[class_descr.index(target_class)] = 1 #create one-hot label vector
#get the base instance, which is a "4" picture
#The image has been normalized!!!
target_instance_tensor = poison_dataset[4200][0]
target_array = target_instance_tensor.numpy().transpose(1,2,0)

#获取基类图像列表，共20张
base_class = '9'
base_label = np.zeros(len(class_descr))
base_label[class_descr.index(base_class)] = 1 #create one-hot label vector
base_data_list = []
for i in range(20):
    base_data_list.append(poison_dataset[9100+i][0])


#测试攻击过程

from poison import FCA2
target_layer = 'stage3.block18'

poison_attack = FCA2.PoisonAttack2(target_instance_tensor,base_data_list,model,target_layer)
poison_images,poison_labels = poison_attack.poison()