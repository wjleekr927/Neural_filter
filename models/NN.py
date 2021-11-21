# NF
import torch
from torch import nn

class NF(nn.Module):
    def __init__(self, channel_taps, device):
        super(NF,self).__init__()
        self.channel_taps = torch.from_numpy(channel_taps).to(device)
        self.CNN_stacks = nn.Sequential(
            nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 1),
            nn.ELU(),
            nn.Conv1d(1,2,1),
            nn.ELU()
        )

    def forward(self, x):
        rst = self.CNN_stacks(x)

        # Reverse the tensor to be applied (multiplied) and reshape it
        channel_taps = torch.flip(self.channel_taps, dims = [0]).reshape(-1,1)
        
        # e.g., a+bj * c+dj = (ac-bd) + (ad+bc)j
        rst_real = torch.mm(rst[:,0,:], channel_taps.real.float()) - torch.mm(rst[:,1,:], channel_taps.imag.float())
        rst_imag = torch.mm(rst[:,0,:], channel_taps.imag.float()) + torch.mm(rst[:,1,:], channel_taps.real.float())

        # Stack to represent complex value
        rst_fin = torch.stack((rst_real,rst_imag), dim = 1)

        return rst_fin