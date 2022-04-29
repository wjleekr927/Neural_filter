# NF
import torch
from torch import nn

class NF(nn.Module):
    def __init__(self, filter_size):
        super(NF,self).__init__()
        self.filter_size = filter_size

        self.TF_enc_layer = nn.TransformerEncoderLayer(d_model = filter_size, dim_feedforward = 2*filter_size, dropout = 0.3, nhead = 4)
        self.TF_enc = nn.TransformerEncoder(self.TF_enc_layer, num_layers = 4)
        self.FC_dec = nn.Sequential(
            nn.Linear(filter_size, 1),
            nn.ELU(),
        )

        self.FC_stacks = nn.Sequential(
            nn.Conv1d(2, 40, kernel_size = 1),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(p = 0.3),
            #nn.MaxPool1d(2, stride = 1),
            # nn.Conv1d(2, 2, kernel_size = 8, stride = 1)
            nn.Linear(filter_size, filter_size // 2),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(p = 0.4),
            nn.Linear(filter_size // 2, 1),
            nn.BatchNorm1d(40),
            nn.ELU(),
            nn.Dropout(p = 0.2),
            nn.Conv1d(40, 2, kernel_size = 1),
            nn.BatchNorm1d(2)
            # nn.GELU(),
            # nn.Dropout(p = 0.3),
        )
        self.linear_embedding = nn.Sequential(
            nn.Conv1d(2, 2, kernel_size = filter_size),
            nn.BatchNorm1d(2)
            #nn.Linear(filter_size, 1)
        )
        self.activ = nn.ELU()

    def forward(self, x):
        # rst = self.TF_enc(x)
        # rst = self.FC_dec(rst)
        rst = self.FC_stacks(x)
        short_cut = self.linear_embedding(x)
        rst = self.activ(short_cut + rst)  
        return rst
