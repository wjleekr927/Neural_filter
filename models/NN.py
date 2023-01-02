# NF
import torch
import math
from torch import nn

class NF(nn.Module):
    def __init__(self, total_taps, RX_num, filter_size, mod_scheme = 'QPSK'):
        super(NF, self).__init__()
        self.filter_size = filter_size
        self.RX_num = RX_num
        self.total_taps = total_taps
        self.mod_scheme = mod_scheme

        self.TF_enc_layer = nn.TransformerEncoderLayer(d_model = filter_size, dim_feedforward = 2*filter_size, dropout = 0.3, nhead = 4)
        self.TF_enc = nn.TransformerEncoder(self.TF_enc_layer, num_layers = 4)
        self.FC_dec = nn.Sequential(
            nn.Linear(filter_size, 1),
            nn.ELU(),
        )

        self.only_FC_stacks = nn.Sequential(
            nn.Linear(filter_size, 2*filter_size),
            nn.GELU(),
            nn.Linear(2*filter_size, filter_size),
            nn.GELU(),
            nn.Linear(filter_size, filter_size // 2),
            nn.GELU(),
            nn.Linear(filter_size // 2, filter_size // 4),
            nn.GELU(),
            nn.Linear(filter_size // 4, 1)
        )

        self.FC_stacks_1 = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size = self.total_taps),
            nn.GELU(),
            nn.Dropout()
        )

        self.FC_stacks_2 = nn.Sequential(
            nn.Conv1d(20, 2, kernel_size = self.filter_size // self.total_taps, stride = 4),
            nn.GELU()
        )
        
        self.FC_fin = nn.Sequential(
            nn.Linear(math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1,\
            (math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 2 + 1),
            nn.GELU(),
            nn.Linear((math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 2 + 1 ,\
            (math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 5 + 1),
            nn.GELU(),
            nn.Linear((math.floor((self.filter_size + 1 - self.total_taps - (self.filter_size // self.total_taps)) /4) + 1) // 5 + 1 , 1)
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
        
        # self.activ = nn.ELU()
        
        self.deconv = nn.Sequential(
            # Size 'M' to 'M + taps - 1'
            nn.ConvTranspose1d(2, 2, kernel_size= self.total_taps),
            nn.GELU(),
            nn.Dropout(p = 0.25)
        )
        
        self.FC_stacks_0_wp = nn.Sequential(
            nn.Linear(self.filter_size, self.filter_size + self.total_taps - 1),
            nn.GELU(),
            nn.Linear(self.filter_size + self.total_taps - 1, self.filter_size + self.total_taps - 1),
            nn.GELU(),
            nn.Dropout(p = 0.2)
        )
        
        self.FC_stacks_1_wp = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size = self.total_taps, padding = 1),
            nn.GELU(),
            #nn.Dropout()
        )

        self.FC_stacks_2_wp = nn.Sequential(
            nn.Conv1d(20, 2, kernel_size = 3),
            nn.GELU()
        )
        
        self.revised_structure = nn.Sequential(
            nn.Linear(self.RX_num * self.filter_size, round(1.5 * (self.filter_size + self.total_taps - 1))),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(round(1.5 * (self.filter_size + self.total_taps - 1)), self.filter_size + self.total_taps - 1),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(2, 20, kernel_size = self.total_taps),
            nn.GELU(),
            nn.Dropout(),
            nn.Conv1d(20, 2, kernel_size = self.filter_size // self.total_taps),
            nn.GELU(),
            nn.Linear(math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1), (math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 2 + 1),
            nn.GELU(),
            nn.Linear((math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 2 + 1, (math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 5 + 1),
            nn.GELU(),
            nn.Linear((math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 5 + 1, 1),
        )
        
        self.revised_structure_manual = nn.Sequential(
            nn.Linear(16, 24),
            nn.GELU(),
            #nn.Dropout(p=0.1),
            nn.Linear(24, 24),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(2, 8, kernel_size = 8),
            nn.GELU(),
            #nn.Dropout(),
            nn.Conv1d(8, 12, kernel_size = 4),
            nn.GELU(),
            nn.Conv1d(12, 16, kernel_size = 4),
            nn.GELU(),
            nn.Conv1d(16, 20, kernel_size = 4),
            nn.GELU(),
            # nn.Linear(12, 8),
            # nn.GELU(),
            # nn.Linear(8, 4),
            # nn.GELU(),
            # nn.Linear(4, 1)
        )
        
        self.classifier_manual = nn.Sequential(
            nn.Linear(160, 100),
            nn.ReLU(),
            nn.Linear(100, 28),
            nn.ReLU(),
            nn.Linear(28, 4)
        )
        
        if self.mod_scheme == 'QPSK':
            self.classifier = nn.Sequential(
                nn.Linear(2, 4)
            )
            
        elif self.mod_scheme == '16QAM':
            self.classifier = nn.Sequential(
                nn.Linear(2, 16)
            )
            
        ### Below is for large ratio in exp 1.
        # nn.Sequential(
        #     nn.Linear(self.RX_num * self.filter_size, round(0.7 * (self.filter_size + self.total_taps - 1))),
        #     nn.BatchNorm1d(2),
        #     nn.GELU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(round(0.7 * (self.filter_size + self.total_taps - 1)), self.filter_size + self.total_taps - 1),
        #     nn.BatchNorm1d(2),
        #     nn.GELU(),
        #     nn.Dropout(p=0.3),
        #     nn.Conv1d(2, 8, kernel_size = self.total_taps),
        #     nn.GELU(),
        #     nn.Dropout(),
        #     nn.Conv1d(8, 2, kernel_size = self.filter_size // self.total_taps),
        #     nn.GELU(),
        #     nn.Linear(math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1), (math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 2 + 1),
        #     nn.GELU(),
        #     nn.Dropout(),
        #     nn.Linear((math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 2 + 1, (math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 6 + 1),
        #     nn.GELU(),
        #     nn.Dropout(),
        #     nn.Linear((math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 6 + 1, (math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 12 + 1),
        #     nn.GELU(),
        #     nn.Dropout(),
        #     nn.Linear((math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 12 + 1, (math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 25 + 1),
        #     nn.GELU(),
        #     nn.Linear((math.floor(self.filter_size - (self.filter_size // self.total_taps) + 1)) // 25 + 1, 1)
        # )
        ###
        
        self.FC_fin_wp = nn.Sequential(
            nn.Linear(self.filter_size, self.filter_size),
            # WO deconv (below)
            #nn.Linear(self.filter_size - self.total_taps + 3, self.filter_size - self.total_taps + 3),
            nn.GELU(),
            nn.Linear(self.filter_size, self.filter_size // 3),
            # WO deconv (below)
            #nn.Linear(self.filter_size - self.total_taps + 3, (self.filter_size - self.total_taps + 3) // 4 + 1),
            nn.GELU(),
            nn.Linear(self.filter_size // 3, 1)
            # WO deconv (below)
            #nn.Linear((self.filter_size - self.total_taps + 3) // 4 + 1, 1)
        )

    def forward(self, x):
        # Transformer
        # rst = self.TF_enc(x)
        # rst = self.FC_dec(rst)

        # FC stacks
        # rst_fin = self.only_FC_stacks(x)
        
        if self.mod_scheme == "QPSK":
            ##########################################
            # # Original code (without padding)
            # rst_1 = self.FC_stacks_1(x)
            # rst_2 = self.FC_stacks_2(rst_1)
            # rst_fin = self.FC_fin(rst_2)
            ##########################################
            
            # Revised code (with padding)
            #rst_0 = self.deconv(x)
            # rst_0 = self.FC_stacks_0_wp(x)
            # rst_1 = self.FC_stacks_1_wp(rst_0)
            # rst_2 = self.FC_stacks_2_wp(rst_1)
            # rst_fin = self.FC_fin_wp(rst_2)
            
            #아래가 최신꺼
            #rst_fin = self.only_FC_stacks(x)
            # For regression
            # rst_fin = self.revised_structure(x)
            #rst_fin = self.revised_structure_manual(x)
            rst_1 = self.revised_structure_manual(x)
            rst_2 = torch.flatten(rst_1, 1)
            rst_fin = self.classifier_manual(rst_2)
            
            # For classification
            # rst_1 = self.revised_structure(x)
            # rst_2 = torch.flatten(rst_1, 1)
            # rst_fin = self.classifier(rst_2)
            
            #short_cut_1 = self.linear_embedding_1(x)
            #rst_1 = self.activ(short_cut_1 + rst_1)
            
            #short_cut_2 = self.linear_embedding_2(rst_1)
            #rst_2 = self.activ(short_cut_2 + rst_2)

            #rst_fin = self.FC_stacks_3(rst_2)

            # if self.filter_size == 8:
            #     short_cut_3 = self.linear_embedding_3_1(rst_2)
            # else:
            #     short_cut_3 = self.linear_embedding_3(rst_2)

            #rst_fin = self.activ(short_cut_3 + rst_fin) 
            
        elif self.mod_scheme == "16QAM":
            ##########################################
            # # Original code (without padding)
            # rst_1 = self.FC_stacks_1(x)
            # rst_2 = self.FC_stacks_2(rst_1)
            # rst_fin = self.FC_fin(rst_2)
            ##########################################
            # ResNet block with padding
            # Revised code (with padding)
            # # rst_0 = self.deconv(x)
            # Original regression (221206)
            # rst_fin = self.revised_structure(x)
            
            # For classification
            rst_1 = self.revised_structure(x)
            rst_2 = torch.flatten(rst_1, 1)
            rst_fin = self.classifier(rst_2)

            # rst_0 = self.FC_stacks_0_wp(x)
            # rst_1 = self.FC_stacks_1_wp(rst_0)
            # rst_2 = self.FC_stacks_2_wp(rst_1)
            # rst_fin = self.FC_fin_wp(rst_2)
            # rst_fin = None
        else:
            raise Exception("Undefined modulation type is used")

        return rst_fin