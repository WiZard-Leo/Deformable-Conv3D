import numpy as np
import torch.nn as nn
from .DeformableBlock import DeformConv3d, DeformBasicBlock, BasicBlock, Flatten
import torch

class DeformVoxResNet(nn.Module):
    def __init__(self, input_shape=(128, 128, 128), num_classes=2, n_filters=32, stride=2, n_blocks=3, 
                 n_flatten_units=None, dropout=0, n_fc_units=128, deform_idx=[], deform_block_idx=[], first_kernel_size=7):
        super(self.__class__, self).__init__()
        
        self.deform_idx = deform_idx
        self.deform_block_idx = deform_block_idx
        
        self.model = nn.Sequential()

        # Convolutional layers
        if 1 in deform_idx:
            self.model.add_module("conv3d_1", DeformConv3d(1, n_filters, kernel_size=first_kernel_size, padding=1, stride=stride))
        else:
            self.model.add_module("conv3d_1", nn.Conv3d(1, n_filters, kernel_size=first_kernel_size, padding=1, stride=stride))
        self.model.add_module("batch_norm_1", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_1", nn.ReLU(inplace=True))
        
        if 2 in deform_idx:
            self.model.add_module("conv3d_2", DeformConv3d(n_filters, n_filters, kernel_size=3, padding=1))
        else:
            self.model.add_module("conv3d_2", nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1))
        self.model.add_module("batch_norm_2", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_2", nn.ReLU(inplace=True))

        if 3 in deform_idx:
            self.model.add_module("conv3d_3", DeformConv3d(n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2))
        else:
            self.model.add_module("conv3d_3", nn.Conv3d(n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2))
        
        if 1 in deform_block_idx:
            self.model.add_module("block_1", DeformBasicBlock(2 * n_filters, 2 * n_filters))
        else:
            self.model.add_module("block_1", BasicBlock(2 * n_filters, 2 * n_filters))
        
        if 2 in deform_block_idx:
            self.model.add_module("block_2", DeformBasicBlock(2 * n_filters, 2 * n_filters))
        else:
            self.model.add_module("block_2", BasicBlock(2 * n_filters, 2 * n_filters))
        self.model.add_module("batch_norm_3", nn.BatchNorm3d(2 * n_filters))
        self.model.add_module("activation_3", nn.ReLU(inplace=True))

        # Additional layers for n_blocks >= 2
        if n_blocks >= 2:
            if 4 in deform_idx:
                self.model.add_module("conv3d_4", DeformConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2))
            else:
                self.model.add_module("conv3d_4", nn.Conv3d(2 * n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2))
            
            if 3 in deform_block_idx:
                self.model.add_module("block_3", DeformBasicBlock(2 * n_filters, 2 * n_filters))
            else:
                self.model.add_module("block_3", BasicBlock(2 * n_filters, 2 * n_filters))
            
            if 4 in deform_block_idx:
                self.model.add_module("block_4", DeformBasicBlock(2 * n_filters, 2 * n_filters))
            else:
                self.model.add_module("block_4", BasicBlock(2 * n_filters, 2 * n_filters))
            self.model.add_module("batch_norm_4", nn.BatchNorm3d(2 * n_filters))
            self.model.add_module("activation_4", nn.ReLU(inplace=True))

        # Additional layers for n_blocks >= 3
        if n_blocks >= 3:
            if 5 in deform_idx:
                self.model.add_module("conv3d_5", DeformConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2))
            else:
                self.model.add_module("conv3d_5", nn.Conv3d(2 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2))
            
            if 5 in deform_block_idx:
                self.model.add_module("block_5", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_5", BasicBlock(4 * n_filters, 4 * n_filters))
            
            if 6 in deform_block_idx:
                self.model.add_module("block_6", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_6", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_5", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_5", nn.ReLU(inplace=True))

        # Additional layers for n_blocks >= 4
        if n_blocks >= 4:
            if 6 in deform_idx:
                self.model.add_module("conv3d_6", DeformConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2))
            else:
                self.model.add_module("conv3d_6", nn.Conv3d(4 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2))
            
            if 7 in deform_block_idx:
                self.model.add_module("block_7", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_7", BasicBlock(4 * n_filters, 4 * n_filters))
            
            if 8 in deform_block_idx:
                self.model.add_module("block_8", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_8", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_6", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_6", nn.ReLU(inplace=True))

        # Compute flatten units dynamically
        if n_flatten_units is None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, *input_shape)
                dummy_output = self.model(dummy_input)
                n_flatten_units = dummy_output.numel()
        print(n_flatten_units)
        
        # self.model.add_module("flatten_1", Flatten())
        # self.model.add_module("fully_conn_1", nn.Linear(n_flatten_units, n_fc_units))
        # self.model.add_module("activation_6", nn.ReLU(inplace=True))
        # self.model.add_module("dropout_1", nn.Dropout(dropout))
        # self.model.add_module("fully_conn_2", nn.Linear(n_fc_units, num_classes))        
        self.flatten_1 = Flatten()
        self.fully_conn_1 = nn.Linear(n_flatten_units, n_fc_units)
        self.activation_6 = nn.ReLU(inplace=True)        
        self.dropout_1 = nn.Dropout(dropout)

        self.fully_conn_2 = nn.Linear(n_fc_units, n_fc_units)
        self.activation_7 = nn.ReLU(inplace=True)        
        self.dropout_2 = nn.Dropout(dropout)

        self.fully_conn_3 = nn.Linear(n_fc_units, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.flatten_1(x)

        x = self.fully_conn_1(x)
        x = self.activation_6(x)        
        x = self.dropout_1(x)

        feature = self.fully_conn_2(x)
        x = self.activation_7(feature)
        x = self.dropout_2(x)

        logits = self.fully_conn_3(x)
        return feature, logits


class SmallDeformVoxResNet(nn.Module):
    def __init__(self, input_shape=(128, 128, 128), num_classes=2, n_filters=32, stride=2, n_blocks=3, 
                 kernel_sizes=[7, 5, 5], conv_dropout=0,
                 n_flatten_units=None, dropout=0, n_fc_units=128, deform_idx=[], deform_block_idx=[]):
        super(self.__class__, self).__init__()
        """
        Args:
        --- deform_idx - (list) - list of indices of convolutional layers to make deformable (starting from 1).
        """
        self.deform_idx = deform_idx
        self.deform_block_idx = deform_block_idx
        
        self.model = nn.Sequential()

        kernel_size = 3 if len(kernel_sizes) < 1 else kernel_sizes[0]
        if 1 in deform_idx:
            self.model.add_module("conv3d_1", DeformConv3d(1, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)) # n * (x/s) * (y/s) * (z/s)
        else:
            self.model.add_module("conv3d_1", nn.Conv3d(1, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)) # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("batch_norm_1", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_1", nn.ReLU(inplace=True))
        
        kernel_size = 3 if len(kernel_sizes) < 2 else kernel_sizes[1]
        if 2 in deform_idx:
            self.model.add_module("conv3d_2", DeformConv3d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)) # n * (x/s) * (y/s) * (z/s)
        else:
            self.model.add_module("conv3d_2", nn.Conv3d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)) # n * (x/s) * (y/s) * (z/s)        
        self.model.add_module("batch_norm_2", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_2", nn.ReLU(inplace=True))

#         1
        kernel_size = 3 if len(kernel_sizes) < 3 else kernel_sizes[2]
        if 3 in deform_idx:
            self.model.add_module("conv3d_3", DeformConv3d(n_filters, 2 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 2n * (x/2s) * (y/2s) * (z/2s)
        else:
            self.model.add_module("conv3d_3", nn.Conv3d(n_filters, 2 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 2n * (x/2s) * (y/2s) * (z/2s)
        #
        if 1 in deform_block_idx:
            self.model.add_module("block_1", DeformBasicBlock(2 * n_filters, 2 * n_filters))
        else:
            self.model.add_module("block_1", BasicBlock(2 * n_filters, 2 * n_filters))
        self.model.add_module("batch_norm_3", nn.BatchNorm3d(2 * n_filters))
        self.model.add_module("activation_3", nn.ReLU(inplace=True))
        self.model.add_module("conv_dropout_1", nn.Dropout3d(conv_dropout))

#         2
        if n_blocks >= 2:
            kernel_size = 3 if len(kernel_sizes) < 4 else kernel_sizes[3]
            if 4 in deform_idx:
                self.model.add_module("conv3d_4", DeformConv3d(2 * n_filters, 2 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 2n * (x/4s) * (y/4s) * (z/4s)
            else:
                self.model.add_module("conv3d_4", nn.Conv3d(2 * n_filters, 2 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 2n * (x/4s) * (y/4s) * (z/4s)
            #
            if 2 in deform_block_idx:
                self.model.add_module("block_2", DeformBasicBlock(2 * n_filters, 2 * n_filters))
            else:
                self.model.add_module("block_2", BasicBlock(2 * n_filters, 2 * n_filters))
            self.model.add_module("batch_norm_4", nn.BatchNorm3d(2 * n_filters))
            self.model.add_module("activation_4", nn.ReLU(inplace=True))
            self.model.add_module("conv_dropout_2", nn.Dropout3d(conv_dropout))

#         3
        if n_blocks >= 3:
            kernel_size = 3 if len(kernel_sizes) < 5 else kernel_sizes[4]
            if 5 in deform_idx:
                self.model.add_module("conv3d_5", DeformConv3d(2 * n_filters, 4 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 4n * (x/8s) * (y/8s) * (z/8s)
            else:
                self.model.add_module("conv3d_5", nn.Conv3d(2 * n_filters, 4 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 4n * (x/8s) * (y/8s) * (z/8s)
            #
            if 3 in deform_block_idx:
                self.model.add_module("block_3", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_3", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_5", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_5", nn.ReLU(inplace=True))
            self.model.add_module("conv_dropout_3", nn.Dropout3d(conv_dropout))

#         4
        if n_blocks >= 4:
            kernel_size = 3 if len(kernel_sizes) < 6 else kernel_sizes[5]
            if 6 in deform_idx:
                self.model.add_module("conv3d_6", DeformConv3d(4 * n_filters, 4 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 4n * (x/16s) * (y/16s) * (z/16s)
            else:
                self.model.add_module("conv3d_6", nn.Conv3d(4 * n_filters, 4 * n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)) # 4n * (x/16s) * (y/16s) * (z/16s)
            #
            if 4 in deform_block_idx:
                self.model.add_module("block_4", DeformBasicBlock(4 * n_filters, 4 * n_filters))
            else:
                self.model.add_module("block_4", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_6", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_6", nn.ReLU(inplace=True))
            self.model.add_module("conv_dropout_4", nn.Dropout3d(conv_dropout))

#         self.model.add_module("max_pool3d_1", nn.MaxPool3d(kernel_size=3)) # (b/2)n * (x/(2^b)sk) * (y/(2^b)sk) * (z/(2^b)sk) ?
        
        # if n_flatten_units is None:
        #     n_flatten_units = (n_blocks) // 2 * 2 * n_filters * np.prod(np.array(input_shape) // (2 ** n_blocks * stride))
        # print(n_flatten_units)

        if n_flatten_units is None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, *input_shape)
                dummy_output = self.model(dummy_input)
                n_flatten_units = dummy_output.numel()
        print(n_flatten_units)
        
        # self.model.add_module("flatten_1", Flatten())
        # self.model.add_module("fully_conn_1", nn.Linear(n_flatten_units, n_fc_units))
        # self.model.add_module("activation_6", nn.ReLU(inplace=True))
        # self.model.add_module("dropout_1", nn.Dropout(dropout))

        # self.model.add_module("fully_conn_2", nn.Linear(n_fc_units, num_classes))

        self.flatten_1 = Flatten()
        self.fully_conn_1 = nn.Linear(n_flatten_units, n_fc_units)
        self.activation_6 = nn.ReLU(inplace=True)        
        self.dropout_1 = nn.Dropout(dropout)

        self.fully_conn_2 = nn.Linear(n_fc_units, num_classes)
            
    def forward(self, x):
        x = self.model(x)
        x = self.flatten_1(x)

        feature = self.fully_conn_1(x)
        x = self.activation_6(feature)
        x = self.dropout_1(x)
        logits = self.fully_conn_2(x)
        return feature, logits

if __name__ == '__main__':
    import torch
    # net = DeformVoxResNet(input_shape=(256,256,256), num_classes=5, n_fc_units=128).to('cuda:0')
    net = SmallDeformVoxResNet(input_shape=(256,256,256), num_classes=5, n_fc_units=128, dropout=0.3, conv_dropout=0.3).to('cuda:0')
    # print(net)
    x = torch.rand((1,1,256,256,256)).to('cuda:0')
    # net(x)

    
    from pytorch_model_summary import summary

    print(summary(net, x, show_input=False, show_hierarchical=False))

