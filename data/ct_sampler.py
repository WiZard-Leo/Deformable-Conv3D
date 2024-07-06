from torch.utils.data import Sampler
from data.dataset_npy import CTDataset, CTSubDataset
import numpy as np


class MultiLabelClassSampler(Sampler):
    """
    指定类别采样器
    """
    def __init__(self, data_source, target_classes):
        self.data_source = data_source
        self.target_classes = target_classes
        
        all_cls_labels = [list(e[:,-1]) for e in data_source.ct_labels]
        self.indices = []
        for iter, labels in enumerate(all_cls_labels):
            for label in labels:
                if label in target_classes:                    
                    self.indices.append(iter)
                    break

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

class BatchDataSampler(Sampler):
    """
    采集数据批次采样器
    """
    def __init__(self, data_source, batch_idx):
        assert batch_idx in [1, 2]

        self.data_source = data_source                        
        all_file_predix = [e[:2] for e in data_source.ct_filenames]
        self.indices = []

        for iter, prefix in enumerate(all_file_predix):  
            if batch_idx == 1:          
                if prefix in ['01','02','03','04','05']:
                    self.indices.append(iter)                
            elif batch_idx == 2:
                if prefix in ['99']:
                    self.indices.append(iter)                

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    


if __name__ == "__main__":   
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ct_dataset = CTDataset()
    # ct_sub_dataset = CTSubDataset(ct_dataset, ratio=0.1)
    ct_sub_dataset = ct_dataset
    
    ct_dataloader = DataLoader(
            dataset=ct_sub_dataset,
            # batch_size=6,
            batch_size=1,
            collate_fn=ct_sub_dataset.collate_fn,
            pin_memory=True,
            # prefetch_factor=8,
            # num_workers=8,
            sampler=MultiLabelClassSampler(data_source=ct_sub_dataset, target_classes=[1])
            )

    for iter, (img, label) in enumerate(tqdm(ct_dataloader)):
        print(len(label))
        print(img[0].shape)
        print(label[0].shape)  
        break              