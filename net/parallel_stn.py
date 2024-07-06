# 降采样版本
# License: BSD
# Author: Ghassen Hamrouni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.ion()   # interactive mode
class STNNet(nn.Module):
    def __init__(self) -> None:
        super(STNNet, self).__init__()
        self.stn1 = STNBlock()
        self.stn2 = STNBlock()
        self.stn3 = STNBlock()
        # self.stn4 = STNBlock()
        # self.stn5 = STNBlock()        

        self.fc = nn.Linear(15,5)
    
    def forward(self,x):
        output1 = self.stn1(x)
        output2 = self.stn2(x)
        output3 = self.stn3(x)
        # output4 = self.stn4(x)
        # output5 = self.stn5(x)

        output = torch.cat((output1,output2,output3),dim=1)
        output = self.fc(output)
        return output        


class STNBlock(nn.Module):    
    def __init__(self):
        super(STNBlock, self).__init__()
        self.conv1 = nn.Conv3d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv3d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout3d()
        self.fc1 = nn.Linear(20*61*61*61, 50)
        self.fc2 = nn.Linear(50, 5)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 10, kernel_size=5),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 4 * 3 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 60 * 60, 32),
            nn.ReLU(True),
            nn.Linear(32, 4 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        # print(xs.shape)
        xs = xs.view(-1, 10 * 60 * 60 * 60)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        # Perform the usual forward pass
        x = F.relu(F.max_pool3d(self.conv1(x), 2))
        x = F.relu(F.max_pool3d(self.conv2_drop(self.conv2(x)), 2))        
        # print(x.shape)
        x = x.view(-1, 20*61*61*61)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x
    

if __name__ == "__main__":
    # net = Net().to('cuda:4')
    # x = torch.rand((4,1,256,256,256)).to('cuda:4')
    # net(x)
    # 1 gpu memory 5486M
    # 2 gpu memory 7774M
    # 4 gpu memory 17752M
    # 6 gpu memory 21800M

    from deconv.DeformableNets import DeformVoxResNet

    import torch
    net = DeformVoxResNet(input_shape=(256,256,256), num_classes=5).to('cuda:7')
    x = torch.rand((4,1,256,256,256)).to('cuda:7')
    net(x)
