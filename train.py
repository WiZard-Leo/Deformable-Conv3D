"""
3D ResDeConv Multi-label Classification Task on CT Project
@Verison:   2024-06-24
@Author:    liuweizhe
"""
# third_party packages
import argparse
import logging
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log')

# custom packages
from net.deconv.DeformableNets import DeformVoxResNet, SmallDeformVoxResNet
from data.dataset_npy import CTDataset, CTSubDataset, CTBalancedDataset
from data.ct_sampler import MultiLabelClassSampler
from utils.losses import MultiLabelCenterLoss, MultiLabelLossWithLabelSmoothing
from utils.utils import EarlyStopping
from utils.utils import calculate_metrics, precision_recall_f1, convert_to_multihot, save_checkpoint, visualize_tsne


num2category = {
    0:"枪支部件",
    1:"枪支",
    2:"刀具",
    3:"打火机",
    4:"充电宝锂电池"
}


"""
Basic Settings
"""
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--model', type=str, default='small_deconv3d')
parser.add_argument('--center_loss', default=False, action='store_true')
parser.add_argument('--conf_thres', type=float, default=0.4)
parser.add_argument('--label_smooth', type=float, default=0.1, help='标签平滑 缓解过拟合')
parser.add_argument('--specific_cls_sample', type=bool, default=True, help='指定类别采样')
parser.add_argument('--num_class', type=int, default=5)
parser.add_argument('--p_dropout', type=float, default=0.3)
parser.add_argument('--p_conv_dropout', type=float, default=0.3)
args = parser.parse_args()
print(args)

device = args.device
conf_thres = args.conf_thres
epochs = args.epochs
model_name = args.model
num_class = args.num_class

logging.basicConfig(filename=f'work_dir/training_{model_name}.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.info(f'参数配置: {args}')


"""
Model Setting model,amp,optimizer,scheduler,loss
"""
# model = DeformVoxResNet(input_shape=(256,256,256), num_classes=num_class).to(device)
model = SmallDeformVoxResNet(input_shape=(256,256,256), num_classes=5, n_fc_units=128, dropout=args.p_dropout, conv_dropout=args.p_conv_dropout).to(device)
scaler = GradScaler()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
# scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.5)

if args.label_smooth:
    criterion = MultiLabelLossWithLabelSmoothing(smoothing=args.label_smooth)
else:
    criterion = nn.BCEWithLogitsLoss()

if args.center_loss:
    center_loss = MultiLabelCenterLoss(num_classes=num_class, feat_dim=128).to(device=device)


"""
Data Setting
"""
train_dataset = CTDataset(mode='train')
test_dataset = CTDataset(mode='test')

# Subset Sampling
# train_dataset = CTSubDataset(train_dataset, ratio=0.01)
# test_dataset = CTSubDataset(test_dataset, ratio=0.01)

# Balance 1:1 Sampling
# total_dataset = CTDataset(mode='train')
# train_dataset = CTBalancedDataset(total_dataset, mode='train')
# test_dataset = CTBalancedDataset(total_dataset, mode='test')

if args.specific_cls_sample:
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, 
                              collate_fn=train_dataset.collate_fn, num_workers=8, 
                              pin_memory=True, prefetch_factor=2, 
                              sampler=MultiLabelClassSampler(data_source=train_dataset, target_classes=[0,1]))
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, 
                             collate_fn=test_dataset.collate_fn, num_workers=8, 
                             pin_memory=True, prefetch_factor=2,
                             sampler=MultiLabelClassSampler(data_source=test_dataset, target_classes=[0,1]))
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, 
                              collate_fn=train_dataset.collate_fn, num_workers=8, 
                              pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, 
                             collate_fn=test_dataset.collate_fn, num_workers=8, 
                             pin_memory=True, prefetch_factor=2)


"""
Training Pipeline
"""
# Store features and labels for t-SNE visualization
train_features = []
train_labels = []
test_features = []
test_labels = []

def train(epoch):
    logging.info(f'Epoch {epoch} is starting')
    model.train()
    train_loss = 0
    train_loss_cernter = 0    
    TP = np.zeros(num_class)
    FP = np.zeros(num_class)
    FN = np.zeros(num_class)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=100)):          
        label = convert_to_multihot(target, num_classes=num_class)
        data = [d.to(device) for d in data]
        data = torch.stack(data)
        target = label.to(device)
        
        optimizer.zero_grad()
        with autocast():
            feature, output = model(data)
            loss = criterion(output, target)

            if args.center_loss:
                loss_center = center_loss(feature, target)
                loss_center = loss_center/(loss_center/loss).detach()
                train_loss_cernter += loss_center

            train_loss += loss

            pred = torch.sigmoid(output)
            pred = (pred > conf_thres).float()
            tp, fp, fn = calculate_metrics(pred, target)
            TP += tp
            FP += fp
            FN += fn

            # Collect features and labels for t-SNE
            train_features.append(feature.cpu().detach().numpy())
            train_labels.append(target.cpu().detach().numpy())
        
        if args.center_loss:
            total_loss = loss + loss_center
        else:
            total_loss = loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                
        writer.add_scalar('classification_loss', loss, batch_idx)
        writer.add_scalar('total_loss', total_loss, batch_idx)
    
    torch.cuda.empty_cache()
    
    train_loss /= len(train_loader.dataset)
    
    if args.center_loss:
        train_loss_cernter /= len(train_loader.dataset)

    precision, recall, f1 = precision_recall_f1(TP, FP, FN)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))

    if args.center_loss:
        print('\nTrain set: Average loss center: {:.4f}'.format(train_loss_cernter))

    for i in range(num_class):
        print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
        logging.info(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')

    overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
    overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
    print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')
    logging.info(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')

    writer.add_scalar('train_precision', overall_precision, epoch)
    writer.add_scalar('train_recall', overall_precision, epoch)

    # scheduler.step()

"""
Testing Pipeline
"""
def test(epoch):
    model.eval()
    test_loss = 0
    test_loss_center = 0

    TP = np.zeros(5)
    FP = np.zeros(5)
    FN = np.zeros(5)
    with torch.no_grad():                
        for iter, (data, target) in enumerate(tqdm(test_loader, ncols=100)):
            label = convert_to_multihot(target, num_classes=num_class)
            data = [d.to(device) for d in data]
            data = torch.stack(data)
            target = label.to(device)
            with autocast():
                feature, output = model(data)
                loss = criterion(output, target).item()
                test_loss += loss

                if args.center_loss:
                    loss_center = center_loss(feature, target).item()
                    test_loss_center += loss_center

            # 对输出应用sigmoid以获得概率
            pred = torch.sigmoid(output)
            pred = (pred > conf_thres).float()
            tp, fp, fn = calculate_metrics(pred, target)
            TP += tp
            FP += fp
            FN += fn            

            test_features.append(feature.cpu().detach().numpy())
            test_labels.append(target.cpu().detach().numpy())

        # loss calculating        
        test_loss /= len(test_loader.dataset)
        
        if args.center_loss:
            test_loss_center /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))

        if args.center_loss:
            print('\nTest set: Average loss center: {:.4f}'.format(test_loss_center))
        
        # metric calculating
        precision, recall, f1 = precision_recall_f1(TP, FP, FN)
        for i in range(num_class):
            print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
            logging.info(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
        overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
        overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
        print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')
        logging.info(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')

        # logging
        writer.add_scalar('test_loss', test_loss, epoch)
        if args.center_loss:
            writer.add_scalar('test_loss_center', test_loss_center, epoch)
        writer.add_scalar('test_precision', overall_precision, epoch)
        writer.add_scalar('test_recall', overall_precision, epoch)
    return test_loss            


if __name__ == '__main__':
    # early_stopping = EarlyStopping(patience=5, delta=0, mode='min', save_path=f'checkpoint/{model_name}_best_model.pth')
    
    for epoch in range(epochs):
        # train and validate
        train(epoch)
        test_loss = test(epoch)

        # check overfit
        # early_stopping(test_loss, model)
        # if early_stopping.early_stop:
        #         print("Early stopping")
        #         logging.info(f"Early Stopping @ epoch {epoch}")
        #         break
        
        # save checkpoint
        if (epoch) % 2 == 0:
            save_checkpoint(epoch, model, optimizer, scaler, checkpoint_path=f'checkpoint/{model_name}_{epoch}.pth')
        
        # Visualize t-SNE after each epoch
        visualize_tsne(train_features, train_labels, epoch, mode='train', model_name=model_name)
        visualize_tsne(test_features, test_labels, epoch, mode='test', model_name=model_name)

        # Clear stored features and labels after visualization
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []