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

from models.Channel_taps import channel_gen

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
    
    # channel_taps: shape = (18,1)
    channel_taps = channel_gen(args.total_taps, args.decay_factor, args.rand_seed)

    if args.filter_type == 'NN':
        print("\n-------------------------------")
        print("Neural filter is used")
        # Parameters are needed to be revised
        model = NF(channel_taps).to(args.device)
    elif args.filter_type == 'Linear':
        print("\n-------------------------------")
        print("Linear filter is used")
        model = LF()
    else:
        raise Exception("Filter type should be 'NN' or 'Linear'")

    # Loss and optimizer setting
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    
    # To save a best model
    best_test_loss = float('inf')

    # Training part
    if args.filter_type == 'NN':
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0

            for batch, (X,y) in enumerate(train_dataloader):
                X, y = X.to(args.device), y.unsqueeze(2).to(args.device)
                pred = model(X)
                loss = loss_fn(pred,y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / (batch+1)
            print("Average loss / epoch = {}".format(epoch_loss)) 
    else:
        pass

    # Testing part

    # np.load( '' + '/symb_len_{}_mod_{}_S_{}'.format(str(symb_len), args.mod_scheme, args.rand_seed) +'.npy')   
    
    # tensorboard --logdir=runs
    # tensorboard --inspect --event_file=myevents.out --tag=loss
    print("\nDone!\n")