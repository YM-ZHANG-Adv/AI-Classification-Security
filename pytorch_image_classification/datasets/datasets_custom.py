from typing import Tuple, Union

import pathlib

import torch
import torchvision
import yacs.config
import cv2
import numpy as np

from torchvision import datasets,transforms

from torch.utils.data import Dataset

from pytorch_image_classification import create_transform

import pandas as pd 


csv_dir = '/root/ZYM/zym/AI_S/torch_classification/data/MNIST/train_mnist_custom.csv' #Change the dir to your own csv file 
csv_dir_test = '/root/ZYM/zym/AI_S/torch_classification/data/MNIST/test_mnist_custom.csv'


def get_label_from_csv(csv_dir = csv_dir):
    df = pd.read_csv(csv_dir)
    label_all = df.iloc[:,1].unique()
    return label_all
    


class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


    
        
class CustomDataset(Dataset):
    def __init__(self,config: yacs.config.CfgNode,is_train: bool,transform = None) -> None:
        if is_train:
            self.csv_path = csv_dir
            self.labels = get_label_from_csv(csv_dir)
            df = pd.read_csv(self.csv_path)
            self.dict_list = df.to_dict(orient='records')
            self.transform = create_transform(config,is_train=True)
        else:
            self.csv_path = csv_dir_test
            self.labels = get_label_from_csv(csv_dir_test)
            df = pd.read_csv(self.csv_path)
            self.dict_list = df.to_dict(orient='records')
            self.transform = create_transform(config,is_train=False)
    
    def __len__(self):
        return len(self.dict_list)
        
    def __getitem__(self, index):
        dic_index = self.dict_list[index]
        label,path = dic_index.values()
        img = cv2.imread(path)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = self.transform(img_tensor)
        img_tensor = img_tensor.permute(1,2,0)
        return (img_tensor,label)
    
    
class CustomValDataset(Dataset):
    def __init__(self,config: yacs.config.CfgNode,is_train: bool,transform = None) -> None:
        csv_dir = config.dataset.dataset_dir + '/'
        if is_train:
            self.csv_path = csv_dir
            self.labels = get_label_from_csv(csv_dir)
            df = pd.read_csv(self.csv_path)
            self.dict_list = df.to_dict(orient='records')
            self.transform = create_transform(config,is_train=True)
        else:
            self.csv_path = csv_dir_test
            self.labels = get_label_from_csv(csv_dir_test)
            df = pd.read_csv(self.csv_path)
            self.dict_list = df.to_dict(orient='records')
            self.transform = create_transform(config,is_train=False)
    
    def __len__(self):
        return len(self.dict_list)
        
    def __getitem__(self, index):
        dic_index = self.dict_list[index]
        label,path = dic_index.values()
        img = cv2.imread(path)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = self.transform(img_tensor)
        img_tensor = img_tensor.permute(1,2,0)
        return (img_tensor,label)
    
    
def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'FashionMNIST',
            'MNIST',
            'KMNIST',
    ]:
        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                return train_dataset, test_dataset
            else:
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=False)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=False)
            return dataset
    elif config.dataset.name == 'ImageNet':
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_dir / 'train', transform=train_transform)
        val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                       transform=val_transform)
        return train_dataset, val_dataset
    
    
    elif config.dataset.name == 'Self_Dataset': 
        print("self_maede_dataset")
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = CustomDataset(config,
                                       is_train=is_train,
                                       transform=train_transform)
                test_dataset = CustomDataset(config,
                                      is_train=False,
                                      transform=val_transform)
                return train_dataset, test_dataset
            else:
                dataset = CustomDataset(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = CustomValDataset(config,
                             is_train=is_train,
                             transform=transform
                             )
            return dataset

    elif config.dataset.name in ['MSTAR']:
        train_set_path = config.dataset.dataset_dir + 'TRAIN/'
        test_set_path = config.dataset.dataset_dir + 'TEST/'
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = datasets.ImageFolder(root = train_set_path,transform= train_transform)
                val_dataset = datasets.ImageFolder(root= test_set_path,transform=val_transform)
                return train_dataset, val_dataset
            else:
                dataset = datasets.ImageFolder(root= train_set_path,transform= None)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = datasets.ImageFolder(root= test_set_path,transform=val_transform)
            return dataset
        
        print("self_defined standar custom dataset")
        
    else:
        raise ValueError('Value of config.dataset.name not found,check model.yaml')
