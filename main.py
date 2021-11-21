# Import packages
import sys
import torch
import numpy as np
import os
import os.path

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
    if args.filter_type == 'NN':
        print("\t[Epoch {}], [Filter type {}], [Filter size {}], [Random seed {}], [Device {}]".\
        format(args.epochs, args.filter_type, args.filter_size, args.rand_seed, args.device))
    elif args.filter_type == 'Linear':
        print("\t[Package {}], [Filter type {}], [Filter size {}], [Random seed {}]".\
        format("CVXPY", args.filter_type, args.filter_size, args.rand_seed))

    if args.mod_scheme == 'QPSK':
        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        symb_len = args.seq_len // bits_per_symb
    else:
        pass
        
    # Build custom dataset (for train and test)
    train_dataset = CustomDataset(symb_len, args.filter_size, args.mod_scheme, args.rand_seed)
    test_dataset = CustomDataset(symb_len, args.filter_size, args.mod_scheme, args.rand_seed, test = True)

    train_dataloader = DataLoader(train_dataset, batch_size = args.bs, drop_last=True, shuffle = True )
    test_dataloader = DataLoader(test_dataset, batch_size = args.bs, drop_last=True, shuffle = True)
    
    # channel_taps: shape = (L,1)
    channel_taps = channel_gen(args.total_taps, args.decay_factor, args.rand_seed)

    if args.filter_type == 'NN':
        print("\n-------------------------------")
        print("Neural filter is used")
        
        # Parameters are needed to be revised
        model = NF(args.filter_size, channel_taps, args.device).to(args.device)
        
        # Loss and optimizer setting
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    
    elif args.filter_type == 'Linear':
        print("\n-------------------------------")
        print("Linear filter is used")
        # model = LF()
    else:
        raise Exception("Filter type should be 'NN' or 'Linear'")

    # To save a best model
    best_test_loss = float('inf')

    # Training part
    if args.filter_type == 'NN':
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0

            for batch, (X,y) in enumerate(train_dataloader):
                X, y = X.to(args.device), y.unsqueeze(2).to(args.device)
                # Channel taps already applied to pred
                pred = model(X)
                loss = loss_fn(pred,y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / (batch+1)
            print("[Epoch {}] Average loss per epoch = {}".format(epoch+1, np.round(epoch_loss,4))) 
    else:
        # After complete the code, change the form as 'model = LF()' 
        import cvxpy as cp
        LF_weight = cp.Variable((args.filter_size, 1), complex = True)
        input_file_name = 'filter_input_len_{}_filter_size_{}_mod_{}_S_{}'.format(symb_len, args.filter_size, args.mod_scheme, args.rand_seed)
        input_file_PATH = './data/symbol_tensor/train_data/' + input_file_name + '.npy'

        target_file_name = 'symb_len_{}_mod_{}_S_{}'.format(symb_len, args.mod_scheme, args.rand_seed)
        target_file_PATH = './data/symbol_tensor/train_data/' + target_file_name + '.npy'        
        
        if os.path.isfile(input_file_PATH):
            TX_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]

        if os.path.isfile(target_file_PATH):
            target_symb = np.load(target_file_PATH)[:,0] + 1j * np.load(target_file_PATH)[:,1]

        total_symb_num = TX_symb.shape[0]

        channel_matrix = np.zeros((total_symb_num,total_symb_num), dtype = 'complex_')

        for idx in range(total_symb_num):
            if idx < args.total_taps:
                channel_matrix[:,idx][:idx+1] = np.flip(channel_taps[: idx+1]).reshape(-1)
            else:
                channel_matrix[:,idx][(idx+1)-args.total_taps : idx+1] = np.flip(channel_taps).reshape(-1)

        objective = cp.Minimize(cp.sum_squares((TX_symb @ LF_weight).T @ channel_matrix - target_symb.reshape(1,-1)))
        prob = cp.Problem(objective)
        # MSE for a single symbol
        opt_MSE_value = prob.solve() / total_symb_num
        print("Optimal train MSE value: {:.4f}".format(opt_MSE_value))

    # Testing part
    if args.filter_type == 'NN':
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch, (X,y) in enumerate(test_dataloader):
                X, y = X.to(args.device), y.unsqueeze(2).to(args.device)
                pred = model(X)
                loss = loss_fn(pred,y)
                test_loss += loss.item()
        test_loss = test_loss / (batch+1)
        print("\nAverage test loss (per {} symbols) = {}".format(X.shape[0], np.round(test_loss,4)))

    else:
        input_file_name = 'filter_input_len_{}_filter_size_{}_mod_{}_S_{}'.format(symb_len, args.filter_size, args.mod_scheme, args.rand_seed)
        input_file_PATH = './data/symbol_tensor/test_data/' + input_file_name + '.npy'

        target_file_name = 'symb_len_{}_mod_{}_S_{}'.format(symb_len, args.mod_scheme, args.rand_seed)
        target_file_PATH = './data/symbol_tensor/test_data/' + target_file_name + '.npy'        
        
        if os.path.isfile(input_file_PATH):
            TX_test_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]

        if os.path.isfile(target_file_PATH):
            target_symb = np.load(target_file_PATH)[:,0] + 1j * np.load(target_file_PATH)[:,1]

        total_symb_num = TX_test_symb.shape[0]
        LF_weight = LF_weight.value

        opt_test_MSE = np.square(np.abs(np.matmul(np.matmul(TX_test_symb, LF_weight).T, channel_matrix) - target_symb.reshape(1,-1))).mean()

        # MSE for a single symbol
        print("Optimal test MSE value: {:.4f}".format(opt_test_MSE))
        
    # tensorboard --logdir=runs
    # tensorboard --inspect --event_file=myevents.out --tag=loss
    print("\nDone!\n")