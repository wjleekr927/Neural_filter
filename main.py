# Import packages
import sys
import torch
import numpy as np

from utils.options import args_parser

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm
from tqdm import trange

# from sklearn.model_selection import train_test_split

from data.custom_set import CustomDataset
from models.NN import NF 
from models.Linear import LF

if __name__ == '__main__':
    # Parse args (from ./utils/options.py)
    args = args_parser()
    
    # Device setting
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Print about current option
    print("\nTraining under the following settings:")
    print("\t[Epoch {}], [Filter type {}], [Filter size {}], [Random seed {}], [Device {}]".\
    format(args.epochs, args.filter_type, args.filter_size, args.rand_seed, args.device))
    
    if args.mod_scheme == 'QPSK':
        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        symb_len = args.seq_len // bits_per_symb
    else:
        pass
        
    # Build custom dataset (for train and test)
    train_dataset = CustomDataset(symb_len, args.mod_scheme, args.rand_seed)
    test_dataset = CustomDataset(symb_len, args.mod_scheme, args.rand_seed, test = True)

    train_dataloader = DataLoader(train_dataset, batch_size = args.bs, drop_last=True, shuffle = True )
    test_dataloader = DataLoader(test_dataset, batch_size = args.bs, drop_last=True, shuffle = True)

    import ipdb; ipdb.set_trace()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)

    # Channel taps apply args.total_taps


    # To save a best model
    best_test_loss = float('inf')

    

    

    # np.load( '' + '/symb_len_{}_mod_{}_S_{}'.format(str(symb_len), args.mod_scheme, args.rand_seed) +'.npy')   
    

    
    
    
    # tensorboard --logdir=runs
    # tensorboard --inspect --event_file=myevents.out --tag=loss
    print("\nDone!\n")