import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
'''Siamese Network with Node Convolution
   Args：num_node (int): Number of regions of interest(ROI)
   
   Shape:
        - Input:  (N, in_channels, ROI_number, ROI_number)
        - Output: (N, 1)
'''
class SNNC(nn.Module):
    def __init__(self, num_node=200):
        super(SNNC, self).__init__()
        self.num_node = num_node
        # row convolution
        self.row = nn.Conv2d(1, 64, (1, self.num_node))
        # column convolution
        self.col = nn.Conv2d(64, 100, (self.num_node, 1))
        self.fc=nn.Linear(100,1)

    def forward_once(self, x):
        out = F.leaky_relu(self.row(x), negative_slope=0.33)
        out = F.leaky_relu(
            self.col(out),
            negative_slope=0.33)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        d12=output1-output2
        d12=self.fc(d12)
        d12=d12.squeeze()
        return d12