import errno # Package to raise an error message
import numpy as np

import os
import os.path

import torch
from torch.utils.data import Dataset, dataloader 
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_length, L, RX_num, filter_size, mod_scheme, decision_delay, seed, test = False):
        # Set the tensor name using arg values
        if test != True:
            input_np_name = 'filter_input_len_{}_RX_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(data_length, RX_num, filter_size, mod_scheme, decision_delay, seed)
            target_np_name = 'filter_target_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(data_length // L, filter_size, mod_scheme, decision_delay, seed)
            
            input_np_PATH = './data/symbol_tensor/train_data/' + input_np_name + '.npy'
            target_np_PATH = './data/symbol_tensor/train_data/' + target_np_name + '.npy'

        else:
            input_np_name = 'filter_input_len_{}_RX_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(data_length, RX_num, filter_size, mod_scheme, decision_delay, seed)
            target_np_name = 'filter_target_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(data_length // L, filter_size, mod_scheme, decision_delay, seed)
            
            input_np_PATH = './data/symbol_tensor/test_data/' + input_np_name + '.npy'
            target_np_PATH = './data/symbol_tensor/test_data/' + target_np_name + '.npy'

        # From the pre-defined path, call the numpy array (if it exists)
        if os.path.isfile(input_np_PATH):
            self.input_data = torch.Tensor(np.load(input_np_PATH))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_np_name)

        if os.path.isfile(target_np_PATH):
            self.target_data = torch.Tensor(np.load(target_np_PATH))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), target_np_name)
    
    # Length of dataset, # of samples
    def __len__(self):
        return len(self.input_data)

    # Take specific sample from the dataset
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.input_data[idx])
        y = torch.FloatTensor(self.target_data[idx])
        return x, y

    # Add str
        '''
    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)
        '''