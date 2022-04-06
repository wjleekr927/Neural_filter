# NF
import torch
from torch import nn

class NF(nn.Module):
    def __init__(self, filter_size):
        super(NF,self).__init__()
        self.filter_size = filter_size
        self.FC_stacks = nn.Sequential(
            nn.Linear(filter_size, filter_size //2),
            nn.GELU(),
            nn.Linear(filter_size //2, 1),
            nn.GELU()
        )

    def forward(self, x):
        rst = self.FC_stacks(x)
        return rst