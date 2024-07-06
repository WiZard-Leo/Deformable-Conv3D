"""
3D ResDeConv Classification Task on CT
cross validation
@Verison:   2024-06-17
@Author:    liuweizhe
@Screen:    
"""
# common packages
import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log')

# custom packages
from net.deconv.DeformableNets import DeformVoxResNet
from data.dataset_npy import CTDataset, CTSubDataset, CTBalancedDataset
from data.ct_sampler import MultiLabelClassSampler
from utils.utils import calculate_metrics, precision_recall_f1, convert_to_multihot, save_checkpoint, visualize_tsne
from utils.losses import MultiLabelLossWithLabelSmoothing, MultiLabelCenterLoss
from utils.utils import EarlyStopping


num2category = {
    0:"枪支部件",
    1:"枪支",
    2:"刀具",
    3:"打火机",
    4:"充电宝锂电池"
}

TestPrecision = np.zeros((4,5))
TestRecall = np.zeros((4,5))

"""
Model Settings
"""
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--learning_rate', type=int, default=1e-2)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--model', type=str, default='deconv_centerloss_tsne')
parser.add_argument('--conf_thres', type=int, default=0.4)
parser.add_argument('--label_smooth', type=bool, default=True)
parser.add_argument('--specific_cls_sample', type=bool, default=True)
args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
conf_thres = args.conf_thres
epochs = args.epochs
model_name = args.model



"""
Data Setting
"""
# skfold = KFold(n_splits=4, shuffle=True)
skfold = StratifiedKFold(n_splits=4, shuffle=True)
full_dataset = CTDataset(mode='all')

# 获取所有标签，用于 KFold 分割
all_labels = np.array([label[0][-1] for label in full_dataset.ct_labels])  
# 获取数据索引
indices = np.arange(len(full_dataset))



def build_model(args):
    """
    Model & Loss Setting
    """
    model = DeformVoxResNet(input_shape=(256,256,256), num_classes=5).to(device)
    # optimizer options
    scaler = GradScaler()
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.5)
    if args.label_smooth:
        criterion = MultiLabelLossWithLabelSmoothing()
    else:
        criterion = nn.BCEWithLogitsLoss()
    center_loss = MultiLabelCenterLoss(num_classes=5, feat_dim=128).to(device=device)

    
    return model, scaler, optimizer, criterion, center_loss
    

"""
Training Pipeline
"""
def train(epoch, train_loader, model, scaler, optimizer, criterion, center_loss):
    # Store features and labels for t-SNE visualization
    train_features = []
    train_labels = []

    model.train()
    train_loss = 0
    train_loss_cernter = 0    
    TP = np.zeros(5)
    FP = np.zeros(5)
    FN = np.zeros(5)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=100)):          
        label = convert_to_multihot(target, num_classes=5)
        data = [d.to(device) for d in data]
        data = torch.stack(data)
        target = label.to(device)
        
        optimizer.zero_grad()
        with autocast():
            feature, output = model(data)            
            loss = criterion(output, target)          

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
        
        total_loss = loss + loss_center
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        writer.add_scalar('classification_loss', loss, batch_idx)
        writer.add_scalar('total_loss', total_loss, batch_idx)
    
    torch.cuda.empty_cache()
    
    train_loss /= len(train_loader.dataset)
    train_loss_cernter /= len(train_loader.dataset)

    precision, recall, f1 = precision_recall_f1(TP, FP, FN)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    print('\nTrain set: Average loss center: {:.4f}'.format(train_loss_cernter))

    for i in range(5):
        print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
    overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
    overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
    print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')

    writer.add_scalar('precision', precision.sum(), epoch)
    writer.add_scalar('recall', recall.sum(), epoch)

    # scheduler.step()
    return train_features, train_labels

"""
Testing Pipeline
"""
def test(epoch, test_loader, model, scaler, optimizer, criterion, center_loss):
    test_features = []
    test_labels = []
    model.eval()
    test_loss = 0
    test_loss_center = 0

    TP = np.zeros(5)
    FP = np.zeros(5)
    FN = np.zeros(5)    
    with torch.no_grad():                
        for iter, (data, target) in enumerate(tqdm(test_loader, ncols=100)):            
            label = convert_to_multihot(target, num_classes=5)
            data = [d.to(device) for d in data]
            data = torch.stack(data)
            target = label.to(device)
            with autocast():
                feature, output = model(data)
                loss = criterion(output, target).item()
                test_loss += loss

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
        test_loss_center /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))
        print('\nTest set: Average loss center: {:.4f}'.format(test_loss_center))
        
        # metric calculating
        precision, recall, f1 = precision_recall_f1(TP, FP, FN)
        for i in range(5):
            print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
        overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
        overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
        print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')

        # logging
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_loss_center', test_loss_center, epoch)
        writer.add_scalar('precision', precision.sum(), epoch)
        writer.add_scalar('recall', recall.sum(), epoch)             
    
    return test_features, test_labels, precision, recall, test_loss


def cross_val():
    # 创建 KFold 对象
    # kf = KFold(n_splits=4)
    kf = StratifiedKFold(n_splits=4, shuffle=True)

    # 交叉验证划分
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, all_labels)):
        print(f'Fold {fold+1}')

        early_stopping = EarlyStopping(patience=5, delta=0, mode='min', save_path='best_model.pth')
        
        # model initialize
        model, scaler, optimizer, criterion, center_loss = build_model(args)
        
        # 创建 Subset 数据集
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # 创建 DataLoader
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, 
                                  collate_fn=CTDataset.collate_fn, num_workers=8,
                                  pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, 
                                collate_fn=CTDataset.collate_fn, num_workers=8,
                                pin_memory=True, prefetch_factor=2)

        # training and validation
        for epoch in range(epochs):
            train_features, train_labels = train(epoch, train_loader, model, scaler, optimizer, criterion, center_loss)
            test_features, test_labels, precision, recall, test_loss = test(epoch, val_loader, model, scaler, optimizer, criterion, center_loss)

            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # Visualize t-SNE after each epoch
            visualize_tsne(train_features, train_labels, epoch, mode='train', fold=fold)
            visualize_tsne(test_features, test_labels, epoch, mode='test', fold=fold)

        TestPrecision[fold] = precision
        TestRecall[fold] = recall
        print(TestPrecision)
        print(TestRecall)


if __name__ == '__main__':
    cross_val()