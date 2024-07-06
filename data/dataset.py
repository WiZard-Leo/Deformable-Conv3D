"""
Name:       CT dataset
Verison:    2024-05-28
CT classification task with 3D Spatial Transformer Networks
"""
import os
import torch
import json
import glob
import pdb
from tqdm import tqdm
import numpy as np
import nibabel as nib
import torch.nn.functional as nnF
import pdb
import random
import logging

from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.transforms import functional as F




category2num = {
    "枪支部件": 0,
    "枪支": 1,
    "刀具": 2,
    "打火机": 3,
    "充电宝锂电池": 4
}


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
    用于分类任务 Spatial Transformer Networks
    """
    def __init__(self, 
                 transforms=data_transform['train'],
                 ct_dataset_dir='/home/ct/data/ctdata/违禁品安检CT数据3通道/data', 
                 ct_dataset_ann='/home/ct/data/ctdata/违禁品安检CT数据/new_label.json',
                 mode='train'):
        super().__init__()
        assert mode in ['train', 'test'], 'dataset must be in ["train", "test"]'
        
        # initialize
        self.transforms = transforms
        self.ct_dataset_ann = ct_dataset_ann
        self.ct_dataset_dir = ct_dataset_dir
        self.mode = mode        
        if self.mode == 'train':
            self.xray_anns = "data/train_dataset.txt"
        else:
            self.mode = 'test'
            self.xray_anns = "data/test_dataset.txt"
        
        
        # set filename and label
        self.ct_filenames = os.listdir(self.ct_dataset_dir)
        self.ct_labels = []
        with open(self.xray_anns, 'r', encoding='utf-8') as read_f:
            for line in read_f.readlines():
                line = line.strip().split(" ")
                filename = line[0]
                label = np.array([int(x) for x in line[2:] if x != '']).reshape(-1, 7)
                for idx in range(label.shape[0]):
                    label[idx][:-1] = [int(_/2) for _ in label[idx][:-1]]            
                if filename in self.ct_filenames:
                    self.ct_labels.append(label)              

    def __len__(self):
        length = len(self.ct_labels)
        return length

    def __getitem__(self, index):
        """
        return CT image of 3chs, Predicted CT image of 3 axises, cls and box label
        [3chs+3axises, height, width, depth]
        """
        # load ct label
        ct_label = self.ct_labels[index]

        # load ct data
        ct_img = self.ct_filenames[index]
        ct_data = nib.load(os.path.join(self.ct_dataset_dir, ct_img))        
        ct_data = ct_data.get_fdata().astype(np.float16)
        ct_data = ct_data[...,-1]
        ct_data = np.expand_dims(ct_data, axis=-1)
        ct_data = np.moveaxis(ct_data, -1, 0)
        ct_data = numpy_to_tensor(ct_data)
        
        img = ct_data
        target = ct_label
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # print(img.shape)
        # [batchsize, chs, 256, 256, 256])
        img = uniform_subsampling(img, factor=2)
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


if __name__ == "__main__":
    # if False:
    #     ct_dataset = CTDataset()
    #     print(len(ct_dataset))
    #     print(ct_dataset[0][0].shape)
    #     exit()

    #     ct_dataloader = DataLoader(
    #         dataset=ct_dataset,
    #         batch_size=2,
    #         shuffle=True,
    #         collate_fn=ct_dataset.collate_fn        
    #         )
        
    #     for iter, (img, label) in enumerate(tqdm(ct_dataloader)):
    #         pdb.set_trace()
    #         print(f'current iteration: {iter}')
    #         print(img[0].shape)
    #         print(label[0].shape)
            
    #         if iter == 5:
    #             break

    ct_dataset = CTDataset()
    ct_sub_dataset = CTSubDataset(ct_dataset, ratio=0.1)
    logging.info(f'the length of sub dataset is {len(ct_sub_dataset)}')
    ct_dataloader = DataLoader(
            dataset=ct_sub_dataset,
            batch_size=2,
            collate_fn=ct_sub_dataset.collate_fn        
            )



    for iter, (img, label) in tqdm(enumerate(ct_dataloader)):
        print(img[0].shape)
        print(label[0].shape)
        pdb.set_trace()
        break