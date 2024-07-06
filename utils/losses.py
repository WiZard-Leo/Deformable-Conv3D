import torch
import torch.nn as nn


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