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

if __name__ == '__main__':
    # Parse args (from ./utils/options.py)
    args = args_parser()

    # 데이터 없으면 에러 띄우기
    # Channel taps apply

    # args.filter_size 가 batchsize

    # Device setting
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    if args.mod_scheme == 'QPSK':
        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        symb_len = args.seq_len // bits_per_symb
        
        
    # np.load( './symbol_tensor' + '/symb_len_{}_mod_{}_S_{}'.format(str(symb_len), args.mod_scheme, args.rand_seed) +'.npy')
    data = CustomDataset()
    
    # Print for current option
    print("\nTraining under following settings:")
    print("\tEpoch: {}, Input M: {}, Model: {}, Device: {}".format(args.epochs, args.M_input, args.model, args.device))
    
    # tensorboard --logdir=runs
    # tensorboard --inspect --event_file=myevents.out --tag=loss
    print("\nDone!\n")