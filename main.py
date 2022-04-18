# Import packages
import errno # Package to raise an error message
import sys
import time
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

from models.Channel import channel_gen
from models.Channel import apply_channel

if __name__ == '__main__':
    # Parse args (from ./utils/options.py)
    args = args_parser()
    
    # Device setting
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Print about current option
    print("\nTraining under the following settings:")
    if args.filter_type == 'NN':
        print("\t[Epochs {}], [Batch size {}], [Filter type {}], [Filter size {}], [Decision delay {}], [SNR {} (dB)], [Random seed (Train) {}], [Random seed (Test) {}], [Device {}]".\
        format(args.epochs, args.bs, args.filter_type, args.filter_size, args.decision_delay, args.SNR, args.rand_seed_train, args.rand_seed_test, args.device))
    elif args.filter_type == 'Linear':
        print("\t[Package {}], [Filter type {}], [Filter size {}], [Random seed (Train) {}], [Random seed (Test) {}]".\
        format("CVXPY", args.filter_type, args.filter_size, args.rand_seed_train, args.rand_seed_test))
    elif args.filter_type == 'LMMSE':
        print("\t[Filter type {}], [Filter size {}], [Decision delay {}], [SNR {} (dB)], [Random seed (Train) {}], [Random seed (Test) {}]".\
        format(args.filter_type, args.filter_size, args.decision_delay, args.SNR, args.rand_seed_train, args.rand_seed_test))

    if args.mod_scheme == 'QPSK':
        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        train_symb_len = args.train_seq_len // bits_per_symb
        test_symb_len = args.test_seq_len // bits_per_symb
    else:
        pass
        
    # channel_taps: shape = (Nc,1) => Follows fixed random seed
    L = args.filter_size + args.total_taps - 1

    # Seed 1111 => easy case, 2077 => default
    channel_seed = 2077
    channel_taps = channel_gen(args.total_taps, args.decay_factor, seed = channel_seed)
    
    train_original_file_name = 'symb_len_{}_mod_{}_S_{}'.format(train_symb_len, args.mod_scheme, args.rand_seed_train)
    train_original_file_PATH = './data/symbol_tensor/train_data/' + train_original_file_name + '.npy'        

    test_original_file_name = 'symb_len_{}_mod_{}_S_{}'.format(test_symb_len, args.mod_scheme, args.rand_seed_test)
    test_original_file_PATH = './data/symbol_tensor/test_data/' + test_original_file_name + '.npy'       
    
    # Symbol shape: (n,)
    if os.path.isfile(train_original_file_PATH):
        train_original_symb = np.load(train_original_file_PATH)[:,0] + 1j * np.load(train_original_file_PATH)[:,1]
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), train_original_file_name)
    
    if os.path.isfile(test_original_file_PATH):
        test_original_symb = np.load(test_original_file_PATH)[:,0] + 1j * np.load(test_original_file_PATH)[:,1]
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), test_original_file_name)

    # Revised to be applied for all cases
    if args.filter_type == "NN" or args.filter_type == "Linear" or args.filter_type == "LMMSE":
        train_input_file_name = '/filter_input_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(train_symb_len, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_train)
        train_input_file_PATH = './data/symbol_tensor/train_data/' + train_input_file_name + '.npy'        

        test_input_file_name = '/filter_input_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(test_symb_len, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_test)
        test_input_file_PATH = './data/symbol_tensor/test_data/' + test_input_file_name + '.npy' 

        # For unseen channel seed
        channel_tap_vector_name = 'channel_{}_taps_S_{}'.format(args.total_taps, channel_seed)
        channel_tap_vector_PATH = './models/channel_tap_vector/' + channel_tap_vector_name + '.npy'

        # 일단 무조건 겪도록 
        if not os.path.isfile(train_input_file_PATH) or not os.path.isfile(test_input_file_PATH) or not os.path.isfile(channel_tap_vector_PATH) or 1:
            # Data generation for NF and LF
            apply_channel(channel_taps, args.filter_size, args.filter_type, train_original_symb, test_original_symb, seed = channel_seed)
    
    # Load the saved channel matrix 
    channel_tap_matrix_PATH = './models/channel_tap_matrix/' + 'channel_matrix_{}_taps_S_{}'.format(args.total_taps, channel_seed) + '.npy'
    channel_matrix = np.load(channel_tap_matrix_PATH)

    if args.filter_type == 'NN':
        # Build custom dataset (for train and test)
        train_dataset = CustomDataset(train_symb_len, L, args.filter_size, args.mod_scheme, args.rand_seed_train)
        test_dataset = CustomDataset(test_symb_len, L, args.filter_size, args.mod_scheme, args.rand_seed_test, test = True)

        train_dataloader = DataLoader(train_dataset, batch_size = args.bs, drop_last = True, shuffle = False )
        test_dataloader = DataLoader(test_dataset, batch_size = args.bs, drop_last = True, shuffle = False)
        print("\n-------------------------------")
        print("Neural filter is used")
        
        # Parameters are needed to be revised
        model = NF(args.filter_size).to(args.device)
        
        # Loss and optimizer setting
        loss_fn = nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    
    elif args.filter_type == 'Linear':
        print("\n-------------------------------")
        print("Linear filter is used")

    elif args.filter_type == 'LMMSE':
        print("\n-------------------------------")
        print("LMMSE filter is used")

    else:
        raise Exception("Filter type should be 'NN' or 'Linear' or 'LMMSE'")

    # To save a best model
    best_test_loss = float('inf')

    #################
    # Training part #
    ################# 
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
            print("[Epoch {:>2}] Average loss per epoch = {:.4f}".format(epoch+1, 2 * epoch_loss))

    elif args.filter_type == 'Linear':
        input_file_name = 'filter_input_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(train_symb_len, args.filter_size, args.mod_scheme, args.decision_delay ,args.rand_seed_train)
        input_file_PATH = './data/symbol_tensor/train_data/' + input_file_name + '.npy'

        target_file_name = 'symb_len_{}_mod_{}_S_{}'.format(train_symb_len, args.mod_scheme, args.rand_seed_train)
        target_file_PATH = './data/symbol_tensor/train_data/' + target_file_name + '.npy'        

        if os.path.isfile(input_file_PATH) and args.filter_type == 'Linear':
            RX_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]
            total_symb_num = RX_symb.shape[0]

        if os.path.isfile(target_file_PATH):
            target_symb = np.load(target_file_PATH)[:,0] + 1j * np.load(target_file_PATH)[:,1]

        model = LF(RX_symb, target_symb, args.filter_size, args.filter_type)
        LF_weight, optimized_rst = model.optimize()

        # MSE for a single symbol
        opt_MSE_value = optimized_rst / total_symb_num
        print("Optimal train MSE value: {:.4f}".format(opt_MSE_value))

    ################
    # Testing part # 
    # LMMSE is implemented only here
    ################
    
    if args.filter_type == 'NN':
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch, (X,y) in enumerate(test_dataloader):
                X, y = X.to(args.device), y.unsqueeze(2).to(args.device)
                pred = model(X)
                loss = loss_fn(pred,y)
                # Complex sign is equal => 1 + 1 = 2, and count this
                correct += torch.sum(torch.sign(pred * y).sum(axis=1) == 2)
                test_loss += loss.item()
        
        # y.shape[0] is batch size
        correct_rate = correct / ((batch+1) * y.shape[0])
        SER = 1 - correct_rate
        print("\nSymbol error rate (SER): {:.2f} %".format(100 * SER))
        
        test_loss = test_loss / (batch+1)
        # Consider that this is complex loss (multiply 2)
        print("\nAverage test loss (per single symbol) = {:.4f}".format(2*test_loss))

        with open('./results/MSE_test_results.txt','a') as f:
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.2f} (%)], [Epochs {}], [Batch size {}], [Filter size {}], [Decision delay {}], [Total taps {}], [SNR {} (dB)], [Train/test seq length {}/{}], [Random seed (Train) {}], [Random seed (Test) {}], [Date {}]"\
            .format(args.filter_type, 2*test_loss, 100*SER, args.epochs, args.bs, args.filter_size, args.decision_delay, args.total_taps, args.SNR, args.train_seq_len, args.test_seq_len, args.rand_seed_train, args.rand_seed_test, time.ctime()))
            
        with open('./results/channel_MSE.txt','a') as f:
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.2f} (%)], [Channel {}], [Filter size {}], [Decision delay {}], [Total taps {}], [Date {}]"\
            .format(args.filter_type, 2*test_loss, 100*SER, channel_taps.T[0], args.filter_size, args.decision_delay, args.total_taps, time.ctime()))

    else:
        input_file_name = 'filter_input_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(test_symb_len, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_test)
        input_file_PATH = './data/symbol_tensor/test_data/' + input_file_name + '.npy'

        if args.filter_type == 'Linear':
            target_file_name = 'symb_len_{}_mod_{}_S_{}'.format(test_symb_len, args.mod_scheme, args.rand_seed_test)
        elif args.filter_type == 'LMMSE':
            target_file_name = 'filter_target_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(test_symb_len // L, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_test)

        target_file_PATH = './data/symbol_tensor/test_data/' + target_file_name + '.npy'        

        if os.path.isfile(target_file_PATH):
            target_symb = np.load(target_file_PATH)[:,0] + 1j * np.load(target_file_PATH)[:,1]

        if os.path.isfile(input_file_PATH) and args.filter_type == 'Linear':
            RX_test_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]
            total_symb_num = RX_test_symb.shape[0]
            opt_test_MSE = np.square(np.abs(np.matmul(RX_test_symb, LF_weight).T - target_symb.reshape(1,-1))).mean()
        
        elif os.path.isfile(input_file_PATH) and args.filter_type == 'LMMSE':
            SNR_ratio = 10**(args.SNR / 10)
            # Solve by matrix calculation
            RX_test_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]
            channel_col = channel_matrix[:,args.decision_delay].reshape(-1,1)

            # LMMSE weight vector // C @ np.conj(C).T = Hermitian & np.linalg.inv(C) = inverse matrix
            # w_LMMSE.shape is (filter_size, 1)
            # w_LMMSE = np.linalg.inv(channel_matrix @ np.conj(channel_matrix).T) @ channel_col

            w_LMMSE = np.linalg.inv(channel_matrix @ np.conj(channel_matrix).T + 1/SNR_ratio * np.eye(args.filter_size)) @ channel_col
            
            opt_test_MSE = 0

            correct = 0
            
            for set_idx in range(RX_test_symb.shape[0]):
                pred_symb = np.conj(w_LMMSE).T @ RX_test_symb[set_idx].reshape(-1,1)  
                if (np.real(target_symb[set_idx]) * np.real(pred_symb) > 0) and (np.imag(target_symb[set_idx]) * np.imag(pred_symb) > 0):
                    correct += 1
                opt_test_MSE += np.square(np.abs(target_symb[set_idx] - pred_symb)).mean()

            SER = (RX_test_symb.shape[0] - correct) / RX_test_symb.shape[0] 
            print("\nSymbol error rate (SER): {:.2f} %".format(100 * SER))
            
            opt_test_MSE /= RX_test_symb.shape[0]

        # MSE for a single symbol
        print("\nOptimal test MSE value (per single symbol): {:.4f}".format(opt_test_MSE))

        with open('./results/MSE_test_results.txt','a') as f:
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.2f} (%)], [Filter size {}], [Decision delay {}], [Total taps {}], [Train/test seq length {}/{}], [Random seed (Train) {}], [Random seed (Test) {}], [Date {}]"\
            .format(args.filter_type, opt_test_MSE, 100 * SER, args.filter_size, args.decision_delay, args.total_taps, args.train_seq_len, args.test_seq_len, args.rand_seed_train, args.rand_seed_test, time.ctime()))
            
        with open('./results/channel_MSE.txt','a') as f:
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.2f} (%)], [Channel {}], [Filter size {}], [Decision delay {}], [Total taps {}], [Date {}]"\
            .format(args.filter_type, opt_test_MSE, 100 * SER, channel_taps.T[0], args.filter_size, args.decision_delay, args.total_taps, time.ctime()))
        
    # tensorboard --logdir=runs
    # tensorboard --inspect --event_file=myevents.out --tag=loss
    print("\n{} filter simulation is DONE!\n".format(args.filter_type))