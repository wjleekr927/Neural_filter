# LF
import torch
from torch import nn

### Correct later ###
class LF(nn.Module):
    def __init__(self):
        super(LF,self).__init__()
        self.CNN_stacks = nn.Sequential(
        )

    def forward(self, x):
        rst = self.CNN_stacks
        return rst