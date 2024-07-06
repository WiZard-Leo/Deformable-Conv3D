"""
3D ResDeConv Classification Task on CT
@Verison:   2024-06-05
@Author:    liuweizhe
@Screen:    710051.wizard.deconv.normal
            865339.wizard.deconv.tsne
"""
import argparse
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

from net.deconv.DeformableNets import DeformVoxResNet, SmallDeformVoxResNet
from data.dataset_npy import CTDataset, CTSubDataset, CTBalancedDataset
from data.ct_sampler import MultiLabelClassSampler


num2category = {
    0:"枪支部件",
    1:"枪支",
    2:"刀具",
    3:"打火机",
    4:"充电宝锂电池"
}


"""
Model Settings
"""
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--learning_rate', type=int, default=1e-2)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--model', type=str, default='deconv_centerloss_tsne')
parser.add_argument('--conf_thres', type=int, default=0.4)
parser.add_argument('--label_smooth', type=bool, default=True)
parser.add_argument('--specific_cls_sample', type=bool, default=False)
parser.add_argument('--num_class', type=int, default=5)
args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
conf_thres = args.conf_thres
epochs = args.epochs
model_name = args.model
num_class = args.num_class
model = DeformVoxResNet(input_shape=(256,256,256), num_classes=num_class).to(device)
# model = SmallDeformVoxResNet(input_shape=(256,256,256), num_classes=5, n_fc_units=128, dropout=0.3, conv_dropout=0.3).to(device)

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
                              sampler=MultiLabelClassSampler(data_source=train_dataset, target_classes=[1]))
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, 
                             collate_fn=test_dataset.collate_fn, num_workers=8, 
                             pin_memory=True, prefetch_factor=2,
                             sampler=MultiLabelClassSampler(data_source=test_dataset, target_classes=[1]))
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, 
                              collate_fn=train_dataset.collate_fn, num_workers=8, 
                              pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, 
                             collate_fn=test_dataset.collate_fn, num_workers=8, 
                             pin_memory=True, prefetch_factor=2)
"""
Loss Setting
"""
scaler = GradScaler()

# optimizer options
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

# scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.5)

class MultiLabelLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.2):
        super(MultiLabelLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + self.smoothing * (1 - targets)
        # Use binary cross-entropy loss
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        return loss

if args.label_smooth:
    criterion = MultiLabelLossWithLabelSmoothing()
else:
    criterion = nn.BCEWithLogitsLoss()


class MultiLabelCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(MultiLabelCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        loss = 0
        for i in range(self.num_classes):
            # Create mask for the current class
            class_mask = labels[:, i].unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Select features of the current class
            class_features = x * class_mask  # Shape: (batch_size, feat_dim)
            
            # Select center of the current class
            class_center = self.centers[i].unsqueeze(0)  # Shape: (1, feat_dim)
            
            # Compute the loss for the current class
            distance = (class_features - class_center) ** 2  # Shape: (batch_size, feat_dim)
            loss += distance.sum() / (class_mask.sum() + 1e-8)
        
        # Average the loss over the batch
        loss /= batch_size
        return loss
    
center_loss = MultiLabelCenterLoss(num_classes=num_class, feat_dim=128).to(device=device)

"""
Utils
"""
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
    """
    index code converts to multihot code
    """
    bz = len(target)
    label = torch.zeros((bz, num_classes))
    for i in range(bz):
        idxes = target[i][...,-1].reshape(-1)
        for idx in idxes:
            label[i][idx] = 1
    return label

"""
Training Pipeline
"""
# Store features and labels for t-SNE visualization
train_features = []
train_labels = []
test_features = []
test_labels = []

def train(epoch):
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

    for i in range(num_class):
        print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
    overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
    overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
    print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')

    writer.add_scalar('train_precision', overall_precision, epoch)
    writer.add_scalar('train_recall', overall_precision, epoch)

    # scheduler.step()

"""
Testing Pipeline
"""
def test():
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
        for i in range(num_class):
            print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
        overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
        overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
        print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')

        # logging
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_loss_center', test_loss_center, epoch)
        writer.add_scalar('test_precision', overall_precision, epoch)
        writer.add_scalar('test_recall', overall_precision, epoch)
        
        
def save_checkpoint(epoch, model, optimizer, scaler, checkpoint_path='checkpoint.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(state, checkpoint_path)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_tsne(features, labels, epoch, mode):
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=np.argmax(labels, axis=1),
        palette=sns.color_palette("hsv", len(num2category)),
        legend="full",
        alpha=0.3
    )
    plt.title(f'{mode} t-SNE visualization at epoch {epoch}')
    plt.savefig(f'visualize/{mode}_tsne_epoch_{epoch}.png')
    plt.close()
    # plt.show()

if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch=epoch)
        test()
        if (epoch) % 2 == 0:
            save_checkpoint(epoch, model, optimizer, scaler, checkpoint_path=f'checkpoint/{model_name}_{epoch}.pth')
        
        # Visualize t-SNE after each epoch
        visualize_tsne(train_features, train_labels, epoch, mode='train')
        visualize_tsne(test_features, test_labels, epoch, mode='test')

        # Clear stored features and labels after visualization
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []