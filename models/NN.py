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

        self.FC_stacks_1 = nn.Sequential(
            nn.Conv1d(2, 2, kernel_size = 1),
            nn.BatchNorm1d(2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(2, 20, kernel_size = 1),
            nn.BatchNorm1d(20),
            nn.Dropout(p = 0.2)
            # nn.Conv1d(2, 2, kernel_size = 8, stride = 1)
            #nn.BatchNorm1d(2)
            #nn.GELU(),
            #nn.Dropout(p = 0.1) 
        )

        self.FC_stacks_2 = nn.Sequential(
            nn.Linear(filter_size, filter_size // 2),
            nn.BatchNorm1d(20),
            nn.GELU(),
            nn.Dropout(p = 0.2),
            nn.Conv1d(20, 10, kernel_size = 1),
            nn.BatchNorm1d(10),
            nn.Dropout(p = 0.2)
        ) 

        self.FC_stacks_3 = nn.Sequential(
            nn.Linear(filter_size // 2, filter_size // 4),
            nn.BatchNorm1d(10),
            nn.GELU(),
            nn.Dropout(p = 0.2),
            nn.Conv1d(10, 2, kernel_size = 1),
            nn.BatchNorm1d(2),
            nn.Dropout(p = 0.2)
        )

        self.linear_embedding_1 = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size = 1)
        )

        self.linear_embedding_2 = nn.Sequential(
            nn.Conv1d(20, 10, kernel_size = 2, stride = 2)
        )

        self.linear_embedding_3 = nn.Sequential(
            nn.Conv1d(10, 2,  kernel_size = 2, stride = 2)
        )

        self.FC_fin = nn.Sequential(
            nn.Linear(filter_size // 4, 1),
            nn.ELU()
        )

        self.activ = nn.ELU()

    # ResNet을 소분하고,
    # Dimension을 조정하기

    def forward(self, x):
        # rst = self.TF_enc(x)
        # rst = self.FC_dec(rst)
        rst_1 = self.FC_stacks_1(x)
        short_cut_1 = self.linear_embedding_1(x)
        rst_1 = self.activ(short_cut_1 + rst_1)

        rst_2 = self.FC_stacks_2(rst_1)
        short_cut_2 = self.linear_embedding_2(rst_1)
        rst_2 = self.activ(short_cut_2 + rst_2)

        rst_fin = self.FC_stacks_3(rst_2)
        short_cut_3 = self.linear_embedding_3(rst_2)
        rst_fin = self.activ(short_cut_3 + rst_fin) 

        rst_fin = self.FC_fin(rst_fin)

        return rst_fin