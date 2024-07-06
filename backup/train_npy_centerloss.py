"""
npy 加速训练版本
256立方
32位float

batchsize 6
numworker 8
prefetch 8
conf threshold 0.3
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from deconv.DeformableNets import DeformVoxResNet
from stn import STNNet
from data.dataset_npy import CTDataset, CTSubDataset, CTBalancedDataset
from net import Net
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log')


import argparse

# Net 1672517.wizard.spatial-transformer.std
#     3014917.wizard.stn.training
# parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:7')
# parser.add_argument('--batchsize', type=int, default=6)
# parser.add_argument('--learning_rate', type=int, default=1e-2)
# parser.add_argument('--epoch', type=int, default=20)
# args = parser.parse_args()

# STNx3 Net 860716.wizard.spatial-transformer.speed_version
# parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:5')
# parser.add_argument('--batchsize', type=int, default=2)
# parser.add_argument('--learning_rate', type=int, default=1e-2)
# parser.add_argument('--epoch', type=int, default=20)
# args = parser.parse_args()

# DeformVoxResNet 1857330.wizard.deconv.test
# DeformVoxResNet 1857330.wizard.deconv.test.upperbound
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--learning_rate', type=int, default=1e-2)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--model', type=str, default='deconv')
args = parser.parse_args()

device = args.device
# model = Net().to(device)
model = DeformVoxResNet(input_shape=(256,256,256), num_classes=5).to(device)
# model = STNNet().to(device)

num2category = {
    0:"枪支部件",
    1:"枪支",
    2:"刀具",
    3:"打火机",
    4:"充电宝锂电池"
}

scaler = GradScaler()

train_dataset = CTDataset(mode='train')
test_dataset = CTDataset(mode='test')

# train_dataset = CTSubDataset(train_dataset, ratio=0.01)
# test_dataset = CTSubDataset(test_dataset, ratio=0.01)

# Balance Setting
# total_dataset = CTDataset(mode='train')
# train_dataset = CTBalancedDataset(total_dataset, mode='train')
# test_dataset = CTBalancedDataset(total_dataset, mode='test')

train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=8, pin_memory=True, prefetch_factor=4)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=8, pin_memory=True, prefetch_factor=4)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
# scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.5)

criterion = nn.BCEWithLogitsLoss()



class MultiLabelCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(MultiLabelCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        centers_batch = self.centers.index_select(0, labels.argmax(dim=1))
        loss = 0
        for i in range(self.num_classes):
            class_mask = labels[:, i].unsqueeze(1)
            class_centers = self.centers[i].unsqueeze(0)
            class_features = x * class_mask
            loss += ((class_features - class_centers) ** 2).sum() / (class_mask.sum() + 1e-8)
        loss /= batch_size
        return loss    
    
center_loss = MultiLabelCenterLoss(num_classes=5, feat_dim=128).to(device=device)

def calculate_metrics(pred, target):
    TP = (pred * target).sum(dim=0).cpu().numpy()
    FP = (pred * (1 - target)).sum(dim=0).cpu().numpy()
    FN = ((1 - pred) * target).sum(dim=0).cpu().numpy()
    return TP, FP, FN


def precision_recall_f1(TP, FP, FN):
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1


def convert_to_multihot(target, num_classes=5):
    bz = len(target)
    label = torch.zeros((bz, num_classes))
    for i in range(bz):
        idxes = target[i][...,-1].reshape(-1)
        for idx in idxes:
            label[i][idx] = 1
    return label

def train(epoch):
    model.train()
    train_loss = 0
    train_loss_cernter = 0
    # total_samples = 0
    TP = np.zeros(5)
    FP = np.zeros(5)
    FN = np.zeros(5)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=100)):  
        # data, target = data.to(device), label.to(device)
        label = convert_to_multihot(target, num_classes=5)
        data = [d.to(device) for d in data]
        data = torch.stack(data)
        target = label.to(device)
        
        optimizer.zero_grad()
        with autocast():
            feature, output = model(data)            
            loss = criterion(output, target)
            loss_center = center_loss(feature, target)
            
            train_loss_cernter += loss_center/(loss_center/loss).detach()
            train_loss += loss

            pred = torch.sigmoid(output)
            pred = (pred > 0.3).float()
            tp, fp, fn = calculate_metrics(pred, target)
            TP += tp
            FP += fp
            FN += fn           
            # total_samples += target.sum().item()
        
        loss = loss+loss_center/(loss_center/loss).detach()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCenter Loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_center.item()))
        
        writer.add_scalar('train_loss', loss, batch_idx)        
        writer.add_scalar('train_loss_cernter', loss_center/(loss_center/loss).detach(), batch_idx)

    
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

def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        test_loss_center = 0

        TP = np.zeros(5)
        FP = np.zeros(5)
        FN = np.zeros(5)
        # total_samples = 0        

        for iter, (data, target) in enumerate(tqdm(test_loader, ncols=100)):
            # data, target = data.to(device), target.to(device)
            label = convert_to_multihot(target, num_classes=5)
            data = [d.to(device) for d in data]
            data = torch.stack(data)
            target = label.to(device)
            with autocast():
                feature, output = model(data)
                test_loss += criterion(output, target).item()
                test_loss_center += center_loss(feature, target).item()

            # 对输出应用sigmoid以获得概率
            pred = torch.sigmoid(output)
            pred = (pred > 0.3).float()
            tp, fp, fn = calculate_metrics(pred, target)
            TP += tp
            FP += fp
            FN += fn
            # correct += (pred * target).sum().item()
            # total_samples += target.sum().item()


        test_loss /= len(test_loader.dataset)
        test_loss_center /= len(test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        #     .format(test_loss, correct, total_samples,
        #             100. * correct / total_samples))
        precision, recall, f1 = precision_recall_f1(TP, FP, FN)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))
        print('\nTest set: Average loss center: {:.4f}'.format(test_loss_center))
        for i in range(5):
            print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
        overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
        overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
        print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_loss_center', test_loss_center, epoch)
        writer.add_scalar('precision', precision.sum(), epoch)
        writer.add_scalar('recall', recall.sum(), epoch)
        
        
def save_checkpoint(epoch, model, optimizer, scaler, checkpoint_path='checkpoint.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(state, checkpoint_path)

if __name__ == '__main__':
    epochs = 10
    for epoch in range(epochs):
        # partial learning

        # feature learning
        train(epoch=epoch)
        test()
        if epoch % 5 == 0:
            save_checkpoint(epoch, model, optimizer, scaler, checkpoint_path=f'{epoch}.pth')
