# NF
import torch
from torch import nn

class NF(nn.Module):
    def __init__(self, channel_taps):
        super(NF,self).__init__()
        self.channel_taps = channel_taps
        self.CNN_stacks = nn.Sequential(
            nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 1),
            nn.ELU(),
            nn.Conv1d(1,2,1),
            nn.ELU()
        )

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        rst = self.CNN_stacks(x)
        # channel_taps: shape = (18,1)
        # Channel_taps should be reversed
        # (8,2,1) 이 되도록 구성
        return rst