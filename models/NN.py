# NF
import torch
from torch import nn

class NF(nn.Module):
    def __init__(self):
        super(NF,self).__init__()
        self.CNN_stacks = nn.Sequential(
            nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 1),
            nn.ELU(),
            nn.Conv1d(1,2,1),
            nn.ELU()
        )

    def forward(self, x):
        rst = self.CNN_stacks(x)
        return rst