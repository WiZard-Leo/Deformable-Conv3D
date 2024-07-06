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
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log')

from net.deconv.DeformableNets import DeformVoxResNet
from data.dataset_npy import CTDataset, CTSubDataset, CTBalancedDataset

num2category = {
    0:"枪支部件",
    1:"枪支",
    2:"刀具",
    3:"打火机",
    4:"充电宝锂电池"
}

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

def load_checkpoint(checkpoint_path, model, optimizer, scaler, device='cuda:0'):
    """
    Load the checkpoint and restore the model, optimizer, and scaler states.
    
    Args:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (torch.nn.Module): The model to load the state into.
    - optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    - scaler (torch.cuda.amp.GradScaler): The scaler to load the state into.
    - device (str): The device to map the loaded state to.
    
    Returns:
    - epoch (int): The epoch at which the checkpoint was saved.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)    
    # Restore the model state
    model.load_state_dict(checkpoint['model_state_dict'])    
    # Restore the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    # Restore the scaler state
    scaler.load_state_dict(checkpoint['scaler_state_dict'])    
    # Get the saved epoch
    epoch = checkpoint['epoch']    
    return model


"""
Model Settings
"""
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--learning_rate', type=int, default=1e-2)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--model', type=str, default='deconv_centerloss_tsne')
parser.add_argument('--conf_thres', type=int, default=0.6)
parser.add_argument('--num_class', type=int, default=5)
parser.add_argument('--checkpoint', type=str, default="/opt/data/private/workplace/deconv3d/checkpoint/deconv_centerloss_tsne_5.pth")
args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
conf_thres = args.conf_thres
epochs = args.epochs
model_name = args.model
num_class = args.num_class
model = DeformVoxResNet(input_shape=(256,256,256), num_classes=num_class).to(device)
test_dataset = CTDataset(mode='test')
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=8, pin_memory=True, prefetch_factor=2)


"""
Loss Setting
"""
scaler = GradScaler()

# optimizer options
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

model = load_checkpoint(checkpoint_path=args.checkpoint, model=model, optimizer=optimizer, scaler=scaler, device=device)

def test():    
    labels = num2category.values()
    model.eval()
    test_loss = 0
    test_loss_center = 0

    y_true = []
    y_pred = []
  
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

            for _ in pred.cpu().numpy().astype(np.int8):
                y_pred.append(_)
            for _ in target.cpu().numpy().astype(np.int8):
                y_true.append(_)
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        print(classification_report(y_true, y_pred, target_names=labels, zero_division=1))

        # loss calculating        
        test_loss /= len(test_loader.dataset)
        test_loss_center /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))
        print('\nTest set: Average loss center: {:.4f}'.format(test_loss_center))
        

if __name__ == '__main__':        
    test()
    