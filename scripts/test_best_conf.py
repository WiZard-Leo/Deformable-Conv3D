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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log')

from net.deconv.DeformableNets import DeformVoxResNet
from data.dataset_npy import CTDataset, CTSubDataset, CTBalancedDataset
from data.ct_sampler import MultiLabelClassSampler
from utils.utils import calculate_metrics, precision_recall_f1, convert_to_multihot, save_checkpoint, visualize_tsne


num2category = {
    0:"枪支部件",
    1:"枪支",
    2:"刀具",
    3:"打火机",
    4:"充电宝锂电池"
}


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--conf_thres', type=int, default=0.6)
parser.add_argument('--num_class', type=int, default=5)
parser.add_argument('--learning_rate', type=int, default=1e-2)
parser.add_argument('--checkpoint', type=str, default="/opt/data/private/workplace/deconv3d/checkpoint/deconv_centerloss_tsne_5.pth")
args = parser.parse_args()

device = args.device
model = DeformVoxResNet(input_shape=(256,256,256), num_classes=5).to(device)
scaler = GradScaler()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate) 


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


def load_test_data(args):    
    test_dataset = CTDataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, 
                            collate_fn=test_dataset.collate_fn, num_workers=8, 
                            pin_memory=True, prefetch_factor=2)
    return test_loader


def test(model, test_loader, conf_thres, device):
    model.eval()

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
                _, output = model(data)
                
            # 对输出应用sigmoid以获得概率
            pred = torch.sigmoid(output)
            pred = (pred > conf_thres).float()
            tp, fp, fn = calculate_metrics(pred, target)
            TP += tp
            FP += fp
            FN += fn            
        
        # metric calculating
        precision, recall, f1 = precision_recall_f1(TP, FP, FN)
        for i in range(5):
            print(f'Class {i} {num2category[i]}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}')
        overall_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
        overall_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-8)
        print(f'Overall: Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}')
        return overall_precision, overall_recall, overall_f1
    

def evaluate_model(model, data, thresholds, device):
    best_threshold = 0.0
    best_f1_score = 0.0
    best_metrics = {}

    for threshold in thresholds:
        precision, recall, f1 = test(model, data, threshold, device)

        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

    return best_threshold, best_metrics


checkpoint_path = '/opt/data/private/workplace/deconv3d/checkpoint/deconv_centerloss_tsne_10.pth'  
model = load_checkpoint(checkpoint_path, model, optimizer, scaler, device)  
test_data = load_test_data(args)

# 定义阈值范围
thresholds = np.linspace(0.1, 0.9, 9)

best_threshold, best_metrics = evaluate_model(model, test_data, thresholds, device)

print(f'Best Threshold: {best_threshold}')
print(f'Precision: {best_metrics["precision"]}')
print(f'Recall: {best_metrics["recall"]}')
print(f'F1-Score: {best_metrics["f1_score"]}')

