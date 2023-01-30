import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class SNNC(nn.Module):
    def __init__(self, num_node=200):
        super(SNNC, self).__init__()
        self.num_node = num_node
        self.row = nn.Conv2d(1, 64, (1, self.num_node))  # [N, 64, H, 1]  输入通道 输出通道  卷积核大小
        self.col = nn.Conv2d(64, 100, (self.num_node, 1))  # [N, 100, 1, 1]
        self.fc=nn.Linear(100,1)

    def forward_once(self, x):
        out = F.leaky_relu(self.row(x), negative_slope=0.33)  # (N, 64, H, 1)
        out = F.leaky_relu(
            self.col(out),
            negative_slope=0.33)  # (N, 100, 1, 1)  negative_slope是x<0时激活函数的斜率
        out = out.view(out.size(0), -1)  # (N, 256)
        return out
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        d12=output1-output2
        d12=self.fc(d12)
      #  d12=self.rl(d12)
        d12=d12.squeeze()
        return d12