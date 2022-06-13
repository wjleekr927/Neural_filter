# NF
import torch
import math
from torch import nn

class NF(nn.Module):
    def __init__(self, total_taps, filter_size):
        super(NF,self).__init__()
        self.filter_size = filter_size
        self.total_taps = total_taps

        self.TF_enc_layer = nn.TransformerEncoderLayer(d_model = filter_size, dim_feedforward = 2*filter_size, dropout = 0.3, nhead = 4)
        self.TF_enc = nn.TransformerEncoder(self.TF_enc_layer, num_layers = 4)
        self.FC_dec = nn.Sequential(
            nn.Linear(filter_size, 1),
            nn.ELU(),
        )

        self.simple_FC_stacks = nn.Sequential(
            nn.Linear(filter_size, filter_size // 2),
            nn.BatchNorm1d(2),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(filter_size // 2, filter_size // 5),
            nn.BatchNorm1d(2),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(filter_size // 5, 1)
        )

        self.only_FC_stacks = nn.Sequential(
            nn.Linear(filter_size, filter_size // 2),
            nn.ELU(),
            nn.Linear(filter_size // 2, filter_size // 5),
            nn.ELU(),
            nn.Linear(filter_size // 5, 1)
        )

        self.FC_stacks_1 = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size = self.total_taps),
            #nn.BatchNorm1d(20),
            nn.GELU(),
            nn.Dropout()
        )

        self.FC_stacks_2 = nn.Sequential(
            nn.Conv1d(20, 2, kernel_size = self.filter_size // self.total_taps, stride = 4),
            #nn.BatchNorm1d(2),
            nn.GELU(),
            #nn.Dropout()
        )

        # self.FC_stacks_3 = nn.Sequential(
        #     nn.Linear(filter_size // 2, filter_size // 5),
        #     nn.BatchNorm1d(8),
        #     nn.ELU(),
        #     nn.Dropout(),
        #     nn.Conv1d(8, 2, kernel_size = 1),
        #     nn.BatchNorm1d(2),
        #     nn.ELU(),
        #     nn.Dropout()
        # )

        # self.linear_embedding_1 = nn.Sequential(
        #     nn.Conv1d(2, 8, kernel_size = 2),
        #     nn.BatchNorm1d(8)
        # )

        # self.linear_embedding_2 = nn.Sequential(
        #     nn.Conv1d(8, 2, kernel_size = 2, stride = 2),
        #     nn.BatchNorm1d(2),
        #     nn.Dropout()
        # )

        # self.linear_embedding_3 = nn.Sequential(
        #     nn.Conv1d(8, 2,  kernel_size = math.floor(filter_size // 5 * .5 + 2), stride = 2),
        #     nn.BatchNorm1d(2)
        # )

        # # For exceptional case
        # self.linear_embedding_3_1 = nn.Sequential(
        #     nn.Conv1d(8, 2,  kernel_size = 3, stride = 2),
        #     nn.BatchNorm1d(2)
        # )

        self.FC_fin = nn.Sequential(
            nn.Linear(math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1,\
            (math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 2 + 1),
            nn.GELU(),
            #nn.Dropout(),
            nn.Linear((math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 2 + 1 ,\
            (math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 5 + 1),
            nn.GELU(),
            #nn.Dropout(),
            # nn.Linear((math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 5 + 1 , \
            # (math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 10 + 1),
            # nn.ELU(),
            nn.Linear((math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 5 + 1 , 1)
        )
        
        # self.FC_fin = nn.Sequential(
        #     nn.Linear(2, 2),
        #     nn.ELU(),
        #     nn.Linear(2, 1)
        # )

        self.activ = nn.ELU()

    def forward(self, x):
        # Transformer
        # rst = self.TF_enc(x)
        # rst = self.FC_dec(rst)

        # FC stacks
        # rst_fin = self.only_FC_stacks(x)

        rst_1 = self.FC_stacks_1(x)
        #short_cut_1 = self.linear_embedding_1(x)
        #rst_1 = self.activ(short_cut_1 + rst_1)

        rst_2 = self.FC_stacks_2(rst_1)
        #short_cut_2 = self.linear_embedding_2(rst_1)
        #rst_2 = self.activ(short_cut_2 + rst_2)

        #rst_fin = self.FC_stacks_3(rst_2)

        # if self.filter_size == 8:
        #     short_cut_3 = self.linear_embedding_3_1(rst_2)
        # else:
        #     short_cut_3 = self.linear_embedding_3(rst_2)

        #rst_fin = self.activ(short_cut_3 + rst_fin) 
        rst_fin = self.FC_fin(rst_2)

        return rst_fin