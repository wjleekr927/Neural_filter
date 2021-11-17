import errno # Package to raise an error message
import numpy as np

import os
import os.path

import torch
from torch.utils.data import Dataset, dataloader 
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    # 수정 필요
    def __init__(self, M_input, M_target, data_length, seed, noise_opt):
        # Set name using arg values
        input_tensor_name = 'input_tensors_len_{}_M_{}_S_{}'.format(data_length, M_input, seed)
        target_tensor_name = 'target_tensors_len_{}_M_{}_S_{}'.format(data_length, M_target, seed)

        input_PATH = 'data/' + input_tensor_name + '.pt'
        target_PATH = 'data/' + target_tensor_name + '.pt'

        # From data folder, call the tensors (if exists)
        if os.path.isfile(input_PATH):
            self.input_data = torch.load(input_PATH)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_tensor_name)
        
        if os.path.isfile(target_PATH):
            self.target_data = torch.load(target_PATH)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), target_tensor_name)

        if noise_opt is True:
            self.input_data = self.input_data + torch.normal(mean = 0, std = .1, size = self.input_data.shape)
            self.target_data = self.target_data + torch.normal(mean = 0, std = .1, size = self.target_data.shape)
            
    # Length of dataset, # of samples
    def __len__(self):
        return len(self.input_data)

    # Take specific sample from dataset
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.input_data[idx])
        y = torch.FloatTensor(self.target_data[idx])
        return x, y

    # Add str
    
        '''
    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)
        '''