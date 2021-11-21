# NF
import torch
from torch import nn

class NF(nn.Module):
    def __init__(self, filter_size, channel_taps, device):
        super(NF,self).__init__()
        self.filter_size = filter_size
        self.channel_taps = torch.from_numpy(channel_taps).to(device)
        self.device = device
        self.CNN_stacks = nn.Sequential(
            nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = filter_size // 4, stride = filter_size // 4),
            nn.ELU(),
            nn.Conv1d(1,2,kernel_size = 4),
            nn.ELU()
        )

    def forward(self, x):

        # Reverse the tensor to be applied (multiplied) and reshape it
        channel_taps = torch.flip(self.channel_taps, dims = [0]).reshape(-1,1)

        rst = self.CNN_stacks(x)
        rst_padding = torch.cat((torch.zeros(len(self.channel_taps)-1,2,1).to(self.device), rst), dim=0)
        rst_fin = torch.zeros(x.shape[0],2,1).to(self.device)

        # Batch size = x.shape[0]
        for idx in range(x.shape[0]):
            # e.g., a+bj * c+dj = (ac-bd) + (ad+bc)j
            rst_real = torch.sum(rst_padding[idx:idx + len(self.channel_taps)][:,0,:] * channel_taps.real.float() \
            - rst_padding[idx:idx + len(self.channel_taps)][:,1,:] * channel_taps.imag.float())
            rst_imag = torch.sum(rst_padding[idx:idx + len(self.channel_taps)][:,0,:] * channel_taps.imag.float() \
            + rst_padding[idx:idx + len(self.channel_taps)][:,1,:] * channel_taps.real.float())

            rst_fin[idx][0], rst_fin[idx][1] = rst_real, rst_imag 

        return rst_fin