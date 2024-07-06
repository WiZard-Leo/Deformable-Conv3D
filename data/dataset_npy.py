"""
CT classification and detection task with 3D Neural Networks
@Verison:   2024-05-30
@Author:    liuweizhe

1. 立方插值256立方
2. 存储为二进制npy格式
3. 64bit --> 32bit
"""
import os
import torch
import random
import logging

import numpy as np
import torch.nn.functional as nnF

from tqdm import tqdm
from torch.utils.data import ConcatDataset, Subset, Dataset, DataLoader, random_split
from torchvision.transforms import functional as F


category2num = {
    "枪支部件": 0,
    "枪支": 1,
    "刀具": 2,
    "打火机": 3,
    "充电宝锂电池": 4
}


# 归一化
def numpy_to_tensor(ct_data):
    """
    Converting 3D ct numpy array to tensor normalized as [0.0,1.0]
    args:
        [channels, height, width, depth]
    """
    ct_data = torch.from_numpy(ct_data)
    for idx in range(ct_data.shape[0]):
        ct_data[idx,...] = (ct_data[idx,...] - torch.min(ct_data[idx,...])) / (torch.max(ct_data[idx,...]) - torch.min(ct_data[idx,...]))    
    return ct_data

# 标准化
def numpy_to_tensor_standardization(ct_data):
    """
    Converting 3D ct numpy array to tensor standardize as [-1.0,1.0]
    args:
        [channels, height, width, depth]
    """
    ct_value_mean = 0.10209798251413127
    ct_value_std = 0.18859518628903846
    #仅在非零区域执行标准化操作
    nonzero_mask = ct_data[...,0] > 0        
    ct_data[...,0] = np.where(nonzero_mask, (ct_data[...,0] - ct_value_mean) / ct_value_std, ct_data[...,0])   
    ct_data = torch.from_numpy(ct_data)
    return ct_data


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        # image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target

def uniform_subsampling(data, factor):
    return data[:,::factor, ::factor, ::factor]


data_transform = {
"train": Compose([ToTensor()]),
"test": Compose([ToTensor()])
}


def convert_to_corner_coordinates(center_x, center_y, width, height):
    left = center_x - width / 2
    top = center_y - height / 2
    right = center_x + width / 2
    bottom = center_y + height / 2
    return left, top, right, bottom


def yolo_to_voc(size, box):
    x = box[0] * size[0]
    w = box[2] * size[0]
    y = box[1] * size[1]
    h = box[3] * size[1]
    xmin = int(x - w/2)
    xmax = int(x + w/2)
    ymin = int(y - h/2)
    ymax = int(y + h/2)
    return (xmin, ymin, xmax, ymax)


class CTDataset(Dataset):
    """
    CT dataset for multi-label classification task and detection mask
    """
    def __init__(self, 
                 transforms=data_transform['train'],
                 ct_dataset_dir='/opt/data/private/workplace/tip3d/my_syn_data',
                 syn_dataset_dir='/opt/data/private/dataset/CT_Database/syndata_npy',
                 mode='train'):
        super().__init__()
        assert mode in ['train', 'test', 'all'], 'dataset must be in ["train", "test" ,"all"]'
        
        # initialize
        self.transforms = transforms        
        self.ct_dataset_dir = ct_dataset_dir
        self.syn_dataset_dir = syn_dataset_dir

        # origin version
        # self.mode = mode        
        # if self.mode == 'train':            
        #     self.xray_anns = "/opt/data/private/dataset/CT_Database/annotations/train_dataset.txt" # 3500
        # elif self.mode == 'test':        
        #     self.xray_anns = "/opt/data/private/dataset/CT_Database/annotations/test_dataset.txt" # 3008
        # elif self.mode == 'all':
        #     self.xray_anns = "/opt/data/private/dataset/CT_Database/annotations/all_dataset.txt" # 6508
        # else:
        #     logging.error(f'unsupport mode: {self.mode}')

        # modified training and testing
        self.mode = mode
        if self.mode == 'train':            
            self.xray_anns = "/opt/data/private/dataset/CT_Database/annotations/syn_dataset_total.txt" # 15300                        
        elif self.mode == 'test':        
            self.xray_anns = "/opt/data/private/dataset/CT_Database/annotations/test_2nd.txt" # 1508
            self.ct_dataset_dir = "/opt/data/private/dataset/CT_Database/ctdata_npy"
            self.syn_dataset_dir = "/opt/data/private/dataset/CT_Database/ctdata_npy"
        elif self.mode == 'all':
            self.xray_anns = "/opt/data/private/dataset/CT_Database/annotations/all_dataset.txt" # 6508
        else:
            logging.error(f'unsupport mode: {self.mode}')
        
        # set filename and label
        self.ct_all_filenames = os.listdir(self.ct_dataset_dir)
        self.ct_syn_filenames = os.listdir(self.syn_dataset_dir)
        self.ct_all_filenames.extend(self.ct_syn_filenames)

        self.ct_filenames = []
        self.ct_labels = []
        with open(self.xray_anns, 'r', encoding='utf-8') as read_f:
            for line in read_f.readlines():
                line = line.strip().split(" ")
                filename = line[0]
                filename = filename[:-7]+'.npy'
                label = np.array([int(x) for x in line[2:] if x != '']).reshape(-1, 7)
                for idx in range(label.shape[0]):
                    label[idx][:-1] = [int(_/2) for _ in label[idx][:-1]]            
                if filename in self.ct_all_filenames:
                    self.ct_labels.append(label)
                    self.ct_filenames.append(filename)

    def __len__(self):
        length = len(self.ct_labels)        
        return length

    def __getitem__(self, index):
        # load ct label
        ct_label = self.ct_labels[index]

        # load ct data
        ct_img = self.ct_filenames[index]
        if os.path.exists(os.path.join(self.ct_dataset_dir, ct_img)):
            ct_data = np.load(os.path.join(self.ct_dataset_dir, ct_img))
        elif os.path.exists(os.path.join(self.syn_dataset_dir, ct_img)):
            ct_data = np.load(os.path.join(self.syn_dataset_dir, ct_img))
        else:
            logging.error('File Not Found')            
        
        # upper bound setting
        ct_mask = np.zeros((256,256,256))
        boxes = ct_label[...,:6]        
        for single_box in boxes:
            ct_mask[single_box[0]:single_box[3],single_box[1]:single_box[4],single_box[2]:single_box[5]] = 1
        ct_data[...,0] = ct_data[...,0] * ct_mask
        # np.save(f'ct_{ct_img}',ct_data)
        # exit()

        # ct_data = np.expand_dims(ct_data, axis=0)
        ct_data = np.moveaxis(ct_data, -1, 0)

        # ct_data = numpy_to_tensor_standardization(ct_data)
        ct_data = torch.from_numpy(ct_data) # replace
        
        img = ct_data
        target = ct_label
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # print(img.shape)
        # [batchsize, chs, 256, 256, 256])
        # img = uniform_subsampling(img, factor=2)
        return img, target
    

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))



class CTSubDataset(Dataset):
    def __init__(self, dataset, ratio):
        self.dataset = dataset

        if ratio:
            random_indexes = random.sample(range(len(self.dataset)), int(ratio * len(self.dataset)))
            self.dataset = Subset(dataset, random_indexes)
            logging.info(f'using a small dataset with ratio: {ratio}')
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        return self.dataset[item]
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
class CTBalancedDataset(Dataset):
    def __init__(self, dataset, mode='train'):        
        # split = [1508,1500]
        split = [2000,1500]
        train_dataset, test_dataset = random_split(dataset, split)
        if mode == 'train':
            self.dataset = train_dataset
        elif mode == 'test':
            self.dataset = test_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        return self.dataset[item]
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == "__main__":  
    ct_dataset = CTDataset()
    # ct_sub_dataset = CTSubDataset(ct_dataset, ratio=0.1)
    ct_sub_dataset = ct_dataset
    logging.info(f'the length of sub dataset is {len(ct_sub_dataset)}')
    ct_dataloader = DataLoader(
            dataset=ct_sub_dataset,
            # batch_size=6,
            batch_size=1,
            collate_fn=ct_sub_dataset.collate_fn,
            pin_memory=True,
            # prefetch_factor=8,
            # num_workers=8,
            )

    for iter, (img, label) in enumerate(tqdm(ct_dataloader)):
        print(len(label))
        print(img[0].shape)
        print(label[0].shape)  
        break              