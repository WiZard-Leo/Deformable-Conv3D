"""
Utils
"""
import torch


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
import numpy as np
import os

num2category = {
    0:"枪支部件",
    1:"枪支",
    2:"刀具",
    3:"打火机",
    4:"充电宝锂电池"
}


def visualize_tsne(features, labels, epoch, mode, model_name):
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
    
    if not os.path.exists(f'visualize/{model_name}'):
        os.mkdir(f'visualize/{model_name}')
    plt.savefig(f'visualize/{model_name}/{mode}_tsne_epoch_{epoch}.png')
    plt.close()
    # plt.show()

class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode='min', save_path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if self.mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.Inf

    def __call__(self, metric, model):
        if self.monitor_op(metric - self.delta, self.best_score):
            self.best_score = metric
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True