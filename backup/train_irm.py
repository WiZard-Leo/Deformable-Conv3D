# Invariant Risk Minimization
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data.dataset_npy import CTDataset, CTSubDataset
from deconv.DeformableNets import DeformVoxResNet
import collections
from utils_invreg import cal_entropy, soft_scl_logits, soft_penalty

def extract_feature_from_gpu(backbone, save_dir):
    train_dataset = CTDataset(mode='train')
    train_dataset = CTSubDataset(train_dataset, ratio=0.001)
    partition_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, 
                                  num_workers=8, pin_memory=True, prefetch_factor=4)
    backbone.eval()

    with torch.no_grad():
        feat_lst = []
        label_lst = []
        for idx, (img, local_labels) in enumerate(tqdm(partition_loader)):
            local_embeddings = backbone(img)
            feat_lst.append(local_embeddings.cpu().numpy())
            
            local_labels = local_labels.cpu().numpy()
            mask = np.zeros((1,1,5,7))            
            for i in range(local_labels.shape[2]):
                for j in range(local_labels.shape[3]):
                    mask[0,0,i,j] = local_labels[0,0,i,j]
            
            label_lst.append(mask)
    feature = np.concatenate(feat_lst, axis=0)
    label = np.concatenate(label_lst, axis=0)
    del feat_lst, label_lst
    del local_labels, local_embeddings
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    np.save(os.path.join(save_dir, 'feature.npy'), feature)
    np.save(os.path.join(save_dir, 'label.npy'), label)

    del partition_loader
    del feature, label
    backbone.train()

class update_split_dataset(Dataset):
    def __init__(self, feature_all, label_all):
        """Initialize and preprocess the dataset."""
        self.feature = torch.from_numpy(feature_all)
        self.label = torch.from_numpy(label_all)

    def __getitem__(self, index):
        """Return one image and its corresponding label."""
        feat = self.feature[index]
        lab = self.label[index]
        return feat, lab

    def __len__(self):
        """Return the number of images."""
        return self.feature.size(0)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
class MLP(nn.Module):
    def __init__(self, head='mlp', dim_in=512, feat_dim=128):
        super(MLP, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        x_norm = F.normalize(x.float(), dim=1)
        mlp_x = self.head(x_norm)
        return mlp_x


def scl_loss(feature, label, temperature=0.3):
    # implementation based on https://github.com/HobbitLong/SupContrast
    base_temperature = temperature
    device = feature.device
    bs = label.shape[0]
    feature = F.normalize(feature, dim=1)

    # create mask
    mask = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(bs).view(-1, 1).to(device), 0)  # non-self mask
    mask = mask * logits_mask  # pos mask

    valid_ind = mask.sum(-1) > 0

    anchor_dot_contrast = torch.div(torch.matmul(feature, feature.T), temperature)[valid_ind]

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask[valid_ind] +1e-8  # both pos and neg logits
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mean_log_prob_pos = (mask[valid_ind] * log_prob).sum(1) / mask[valid_ind].sum(1)

    loss = - (temperature / base_temperature) * mean_log_prob_pos

    return loss.mean()

def train_mlp(mlp_net, train_loader):
    mlp_net.train()
    mlp_optimizer = torch.optim.Adam(list(mlp_net.parameters()), lr=0.2, weight_decay=0.)
    mlp_scheduler = MultiStepLR(mlp_optimizer, [2, 4], gamma=0.1, last_epoch=-1)

    global_step = 0
    total_epoch = 5
    for epoch in tqdm(range(total_epoch)):
        # train_loader.sampler.set_epoch(epoch)
        for ori_feat, label in train_loader:
            bs = label.size(0)
            re_shuffle = torch.randperm(bs)
            ori_feat = ori_feat[re_shuffle].to('cuda:4')
            label = label[re_shuffle].to('cuda:4')
            feat = mlp_net(ori_feat)  # mapped features
            scl_mlp = scl_loss(feat, label)

            mlp_optimizer.zero_grad()
            scl_mlp.backward()
            mlp_optimizer.step()
            lr = mlp_optimizer.param_groups[0]['lr']
            mlp_scheduler.step(epoch=epoch)
            global_step += 1

    return mlp_net


class Partition(nn.Module):
    def __init__(self, n_cls, n_env):
        super(Partition, self).__init__()
        self.partition_matrix = nn.Parameter(torch.randn((n_cls, n_env)))
        self.n_env = n_env

    def forward(self, label):
        sample_split = F.softmax(self.partition_matrix[label], dim=-1)
        return sample_split
    

def scl_loss_mid(feature, label, temperature=0.3):
    # logits: similarity matrix
    # logits_mask: non-self mask
    # mask: same id non-self mask
    # index sequence: [[0~bs], [0~bs]...]

    device = feature.device
    bs = label.shape[0]
    feature = F.normalize(feature, dim=1)

    # create mask
    mask = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(bs).view(-1, 1).to(device), 0)
    mask = mask * logits_mask

    logits = torch.div(torch.matmul(feature, feature.T), temperature)

    # compute the index
    index_sequence = torch.arange(bs).to(device)
    index_sequence = index_sequence.unsqueeze(0).expand(bs, bs)

    valid_ind = mask.sum(-1)>0

    return logits[valid_ind], logits_mask[valid_ind], mask[valid_ind], index_sequence[valid_ind]


def scl_logits(logits, logits_mask, mask):
    assert min(mask.sum(-1))>0
    # for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # both pos and neg logits
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - mean_log_prob_pos

    return loss.mean()


def auto_split_offline(mlp_net, trainloader, part_module, device):    
    irm_mode = 'v2'
    num_env = 2
    constrain=True
    temperature = 0.3
    loss_mode = 'v2'
    nonorm = False
    irm_weight = 0.2
    cons_relax = False
    irm_temp = 0.5

    # irm mode: v1 is original irm; v2 is variance
    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0

    # optimizer and schedule
    part_module.train()
    pre_optimizer = torch.optim.Adam(list(part_module.parameters()), lr=0.5, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [3, 6, 9, 12], gamma=0.2, last_epoch=-1)

    global_step = 0
    total_epoch = 15
    for epoch in range(total_epoch):
        trainloader.sampler.set_epoch(epoch)

        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [], [], [], [], 0

        for ori_feat, label in trainloader:
            global_step += 1
            bs = ori_feat.size(0)
            training_num += bs
            loss_cont_list, loss_penalty_list = [], []

            re_shuffle = torch.randperm(bs)
            ori_feat = ori_feat[re_shuffle].to(device)
            label = label[re_shuffle].to(device)

            with torch.no_grad():
                feat = mlp_net(ori_feat)  # mapped features
            sample_split = part_module(label)

            if irm_mode == 'v1':  # gradient
                for env_idx in range(num_env):
                    logits, logits_mask, mask, index_sequence = scl_loss_mid(feat, label, temperature=1.0)
                    loss_weight = sample_split[:, env_idx][index_sequence]  # [bs, bs-1]
                    cont_loss_env = soft_scl_logits(logits / temperature, logits_mask, mask, loss_weight,
                                                    mode=loss_mode, nonorm=nonorm)
                    loss_cont_list.append(cont_loss_env)

                    # irm_loss based on weighted CL loss
                    penalty_grad = soft_penalty(logits, logits_mask, mask, loss_weight, loss_mode, nonorm, irm_temp)
                    loss_penalty_list.append(penalty_grad)

                cont_loss_epoch = torch.stack(loss_cont_list).mean()  # contrastive loss
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()  # gradient of the CL loss
                risk_final = - (cont_loss_epoch + irm_weight * inv_loss_epoch)

            elif irm_mode == 'v2':  # variance
                for env_idx in range(num_env):
                    logits, logits_mask, mask, index_sequence = scl_loss_mid(feat, label, temperature=1.0)
                    loss_weight = sample_split[:, env_idx][index_sequence]  # [bs, bs-1]
                    cont_loss_env = soft_scl_logits(logits / temperature, logits_mask, mask, loss_weight,
                                                    mode=loss_mode, nonorm=nonorm)
                    loss_cont_list.append(cont_loss_env)

                cont_loss_epoch = torch.stack(loss_cont_list).mean()  # contrastive loss
                inv_loss_epoch = torch.var(torch.stack(loss_cont_list))  # variance of the CL loss
                risk_final = - (cont_loss_epoch + irm_weight * inv_loss_epoch)

            if constrain:  # constrain for balanced partition
                if nonorm:
                    constrain_loss = 0.2 * (- cal_entropy(sample_split.mean(0), dim=0) +
                                            cal_entropy(sample_split, dim=1).mean())
                else:
                    if cons_relax:
                        constrain_loss = torch.relu(0.6365 - cal_entropy(sample_split.mean(0), dim=0))
                    else:
                        constrain_loss = - cal_entropy(sample_split.mean(0),
                                                       dim=0)
                risk_final += constrain_loss

            pre_optimizer.zero_grad()
            risk_final.backward()
            pre_optimizer.step()
            soft_split_print = part_module.module.partition_matrix.detach().clone()

            
            risk_all_list.append(risk_final.item())
            risk_cont_all_list.append(-cont_loss_epoch.item())
            risk_penalty_all_list.append(-inv_loss_epoch.item())
            risk_constrain_all_list.append(constrain_loss.item())

            avg_risk = sum(risk_all_list) / len(risk_all_list)  # total loss
            avg_cont_risk = sum(risk_cont_all_list) / len(risk_cont_all_list)  # CL env
            avg_irm_risk = sum(risk_penalty_all_list) / len(risk_penalty_all_list)  # IRM penalty
            avg_cst_risk = sum(risk_constrain_all_list) / len(risk_constrain_all_list)  # balance penalty

            lr = pre_optimizer.param_groups[0]['lr']
            
            np.save(os.path.join('./', f'partition_{epoch}.npy'),
                    soft_split_print.cpu().numpy())

            if global_step % 200 == 0:
                logging.info(
                    '\rUpdating Env [%d/%d]'
                    'Total_loss: %.2f  '
                    'CL_loss: %.2f  '
                    'Penalty_loss: %.2f  '
                    'Constrain_loss: %.2f  '
                    'Lr: %.4f  Inv_Mode: %s  Soft Split: %s'
                    % (epoch, total_epoch,
                        avg_risk,
                        avg_cont_risk,
                        avg_irm_risk,
                        avg_cst_risk,
                        lr, irm_mode, F.softmax(soft_split_print, dim=-1)))

        pre_scheduler.step()

    soft_split_final = part_module.module.partition_matrix.detach().clone()
    
    logging.info(
        'Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Lr: %.4f  Inv_Mode: %s'
        % (epoch, total_epoch, training_num, len(trainloader.dataset),
            avg_risk,
            avg_cont_risk,
            avg_irm_risk,
            avg_cst_risk,
            lr, irm_mode))
    final_split_softmax = F.softmax(soft_split_final, dim=-1)
    group_assign = final_split_softmax.argmax(dim=1)
    dstr = collections.Counter(group_assign.cpu().numpy().tolist())
    dstr = {key: dstr[key] for key in sorted(dstr)}
    dstr = list(dstr.values())
    logging.info('Distribution:' + ' / '.join([str(d) for d in dstr]))
    del pre_optimizer, final_split_softmax, part_module, soft_split_print
    logging.info('The partition learning is completed, saving best partition matrix...')
    np.save(os.path.join('./', 'final_partition.npy'), soft_split_final.cpu().numpy())

    return soft_split_final


def update_partition(emb, lab, save_dir):
    dataset = update_split_dataset(emb, lab.astype(int))
    train_loader = DataLoader(dataset, batch_size=4, pin_memory=True, num_workers=4, prefetch_factor=4)

    mlp_net = MLP(head='mlp', dim_in=512, feat_dim=128).to('cuda:4')
    mlp_net.apply(weights_init)    
    mlp_net = train_mlp(mlp_net, train_loader)
    mlp_net.eval()

    n_cls = 5
    env_num = 2
    part_module = Partition(n_cls, env_num).cuda()    
    part_module.train().cuda()
    updated_split = auto_split_offline(mlp_net, train_loader, part_module)
    del dataset, train_loader
    del mlp_net

    return updated_split


def partial_learning(backbone, epoch, device):
    updated_split_all = []
    save_dir = os.path.join('saved_feat', 'epoch_{}'.format(epoch))
    extract_feature_from_gpu(backbone, save_dir)

    emb = np.load(os.path.join(save_dir, 'feature.npy'))
    lab = np.load(os.path.join(save_dir, 'label.npy'))
    # conduct partition learning
    logging.info('Started partition learning...')

    updated_split = update_partition(emb, lab, save_dir)
    del emb,lab
    updated_split_all.append(updated_split)


if __name__ == "__main__":
    device = 'cuda:4'
    net = DeformVoxResNet(input_shape=(256,256,256), num_classes=5)
    for epoch in range(20):
        partial_learning(backbone=net, epoch=epoch, device=device)
        break