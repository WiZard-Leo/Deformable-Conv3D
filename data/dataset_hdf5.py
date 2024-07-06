"""

"""
import os
import torch
import json
import glob
import numpy as np
# import nibabel as nib
import torch.nn.functional as nnF
import pdb
import random
import logging
import h5py

from tqdm import tqdm
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


class CTDataset(Dataset):
    """
    CT Dataset
    """
    def __init__(self, 
                 transforms=data_transform['train'],
                 ct_dataset_dir='/home/ct/data/ctdata/total_channel_3/data_hdf5', 
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
            h5_data = h5py.File(os.path.join(ct_dataset_dir, 'train.h5'))
            self.ct_image = h5_data['images']
            self.ct_label = h5_data['labels']
        else:
            self.mode = 'test'
            self.xray_anns = "data/test_dataset.txt"
            h5_data = h5py.File(os.path.join(ct_dataset_dir, 'val.h5'))
            self.ct_image = h5_data['images']
            self.ct_label = h5_data['labels']        

    def __len__(self):
        length = len(self.ct_label.keys())
        return length

    def __getitem__(self, index):        
        ct_label = self.ct_label[str(index)]      
        ct_data = self.ct_image[str(index)]        
        ct_data = np.array(ct_data, dtype=np.float16)
        ct_label = np.array(ct_label, dtype=np.float16)
        ct_data = np.moveaxis(ct_data, -1, 0)
        ct_data = numpy_to_tensor(ct_data)        
        img = ct_data
        target = ct_label
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
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


def main():
    ct_dataset = CTDataset()
    # ct_dataset = CTSubDataset(dataset=ct_dataset,ratio=0.001)
    print(len(ct_dataset))    

    ct_dataloader = DataLoader(
        dataset=ct_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=ct_dataset.collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
        )
    
    for iter, (img, label) in enumerate(tqdm(ct_dataloader)):
        
        print(f'current iteration: {iter}')
        print(img[0].shape)
        print(label[0].shape)
if __name__ == "__main__":
    main()
        
    # ct_dataset = CTDataset()
    # ct_sub_dataset = CTSubDataset(ct_dataset, ratio=0.1)
    # logging.info(f'the length of sub dataset is {len(ct_sub_dataset)}')
    # ct_dataloader = DataLoader(
    #         dataset=ct_sub_dataset,
    #         batch_size=2,
    #         collate_fn=ct_sub_dataset.collate_fn        
    #         )

    # print("hdf5 dataset")

    # for iter, (img, label) in tqdm(enumerate(ct_dataloader)):
    #     print(img[0].shape)
    #     print(label[0].shape)
    #     # break