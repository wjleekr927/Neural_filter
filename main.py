# Import packages
import errno # Package to raise an error message
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path

from utils.options import args_parser
from utils.R_corr import R_calc

from torch import nn
from torch.autograd import Variable
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
    elif args.filter_type == 'LS':
        print("\t[Filter type {}], [Filter size {}], [Decision delay {}], [SNR {} (dB)], [Random seed (Train) {}], [Random seed (Test) {}]".\
        format(args.filter_type, args.filter_size, args.decision_delay, args.SNR, args.rand_seed_train, args.rand_seed_test))

    if args.mod_scheme == 'QPSK':
        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        train_symb_len = args.train_seq_len // bits_per_symb
        test_symb_len = args.test_seq_len // bits_per_symb
        
        norm_cof = np.round(1 / np.sqrt(2), 4)
        QPSK_label_GT_list = [[[+norm_cof], [+norm_cof]], [[+norm_cof], [-norm_cof]], [[-norm_cof], [+norm_cof]], [[-norm_cof], [-norm_cof]]]

    else:
        pass
        
    # channel_taps: shape = (Nc,1) => Follows fixed random seed
    L = args.filter_size + args.total_taps - 1

    # Seed 1111 => easy case, 2077 => default
    channel_seed = args.rand_seed_channel
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
    if args.filter_type == "NN" or args.filter_type == "Linear" or args.filter_type == "LMMSE" or args.filter_type == "LS":
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
        train_dataset = CustomDataset(train_symb_len, L, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_train)
        test_dataset = CustomDataset(test_symb_len, L, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_test, test = True)

        train_dataloader = DataLoader(train_dataset, batch_size = args.bs, drop_last = True, shuffle = False )
        test_dataloader = DataLoader(test_dataset, batch_size = args.bs, drop_last = True, shuffle = False)
        print("\n-------------------------------")
        print("Neural filter is used")

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                #nn.init.kaiming_uniform_(m.weight)
                #nn.init.xavier_normal_(m.weight)
                #nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                #nn.init.kaiming_uniform_(m.weight) 
                #nn.init.xavier_normal_(m.weight) 
                #nn.init.xavier_uniform_(m.weight)  
                #m.bias.data.fill_(.01)

        def one_hot_label_return(origin_tensor):
            rst = torch.where(torch.logical_and(origin_tensor[:,0] > 0, origin_tensor[:,1] > 0), 0, 0)
            init = torch.zeros_like(rst)
            rst += torch.where(torch.logical_and(origin_tensor[:,0] > 0, origin_tensor[:,1] < 0), 1, init)
            rst += torch.where(torch.logical_and(origin_tensor[:,0] < 0, origin_tensor[:,1] > 0), 2, init)
            rst += torch.where(torch.logical_and(origin_tensor[:,0] < 0, origin_tensor[:,1] < 0), 3, init)
            return rst

        # Parameters are needed to be revised
        model = NF(args.total_taps, args.filter_size).to(args.device)
        model.apply(init_weights)
        
        # Loss and optimizer setting
        loss_fn_MSE = nn.MSELoss()
        loss_fn_CE = nn.CrossEntropyLoss()
        loss_fn_MRL = nn.MarginRankingLoss()
        loss_fn_TML = nn.TripletMarginLoss()

        #optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 0.02)

        ###############################################
        import math
        from torch.optim.lr_scheduler import _LRScheduler

        class CosineAnnealingWarmUpRestarts(_LRScheduler):
            def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
                if T_0 <= 0 or not isinstance(T_0, int):
                    raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
                if T_mult < 1 or not isinstance(T_mult, int):
                    raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
                if T_up < 0 or not isinstance(T_up, int):
                    raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
                self.T_0 = T_0
                self.T_mult = T_mult
                self.base_eta_max = eta_max
                self.eta_max = eta_max
                self.T_up = T_up
                self.T_i = T_0
                self.gamma = gamma
                self.cycle = 0
                self.T_cur = last_epoch
                super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
            
            def get_lr(self):
                if self.T_cur == -1:
                    return self.base_lrs
                elif self.T_cur < self.T_up:
                    return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
                else:
                    return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                            for base_lr in self.base_lrs]

            def step(self, epoch=None):
                if epoch is None:
                    epoch = self.last_epoch + 1
                    self.T_cur = self.T_cur + 1
                    if self.T_cur >= self.T_i:
                        self.cycle += 1
                        self.T_cur = self.T_cur - self.T_i
                        self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
                else:
                    if epoch >= self.T_0:
                        if self.T_mult == 1:
                            self.T_cur = epoch % self.T_0
                            self.cycle = epoch // self.T_0
                        else:
                            n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                            self.cycle = n
                            self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                            self.T_i = self.T_0 * self.T_mult ** (n)
                    else:
                        self.T_i = self.T_0
                        self.T_cur = epoch
                        
                self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
                self.last_epoch = math.floor(epoch)
                for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                    param_group['lr'] = lr
        ###############################################
        
        scheduler_gamma = 0.7
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = (args.epochs // 4 + 1), gamma = scheduler_gamma)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = scheduler_gamma)
        # T_0 : 초기 설정하는 주기, T_mult : 그 이후로 얼마나 주기를 늘릴 것인지, eta_min : minimum learning rate
        # 그냥 일단 gamma 변수를 쓰는 중 (eta_min 으로)
        
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = args.epochs // 8, T_mult = 3, eta_min = scheduler_gamma)
        #scheduler_type = type(scheduler).__name__

        # Set small LR to make warm up
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = args.epochs // 9, T_mult = 1, eta_max = 1e-2, T_up = 10, gamma = scheduler_gamma)
        scheduler_type = "Custom Cosine"
    
    elif args.filter_type == 'Linear':
        print("\n-------------------------------")
        print("Linear filter is used")

    elif args.filter_type == 'LMMSE':
        print("\n-------------------------------")
        print("LMMSE filter is used")

    elif args.filter_type == 'LS':
        print("\n-------------------------------")
        print("LS filter is used")

    else:
        raise Exception("Filter type should be 'NN' or 'Linear' or 'LMMSE'")

    #################
    # Training part #
    ################# 
    if args.filter_type == 'NN':
        model.train()

        best_train_model_name = 'model_state_dict.pt'
        best_train_model_PATH = './models/params/' + best_train_model_name
        best_epoch, best_epoch_loss = 0, float('inf')

        for epoch in range(args.epochs):
            epoch_loss = 0
            
            for batch, (X,y) in enumerate(train_dataloader):
                X, y = X.to(args.device), y.unsqueeze(2).to(args.device)
                # Channel taps already applied to pred
                pred = model(X)
                
                # Add detection loss
                if args.mod_scheme == 'QPSK':
                    num_classes = 4
                    pred_softmax = torch.tensor([]).to(args.device)


                for idx, symb_GT in enumerate(QPSK_label_GT_list):
                    QPSK_label_GT = torch.tensor(symb_GT).expand(args.bs, -1, -1).to(args.device)

                    # Distance calculation
                    L2_distance = torch.sqrt(torch.square(pred-QPSK_label_GT).sum(dim = 1))
                    pred_softmax = torch.cat((pred_softmax, torch.exp(-L2_distance)), dim = 1)

                # Normalize like softmax function
                # pred_softmax = pred_softmax / pred_softmax.sum(dim = 1).reshape(-1,1)

                # base_constraint = torch.exp(-4 * torch.ones_like(pred_softmax)).to(args.device)

                base_constraint = 3 * norm_cof * torch.ones_like(pred).to(args.device)
                
                y_one_hot = F.one_hot(one_hot_label_return(y), num_classes = num_classes).float()
                pred_one_hot = F.one_hot(one_hot_label_return(pred), num_classes = num_classes).float()

                # one_hot_loss_weight = 5
                # distance_loss_weight = 0

                #+ loss_fn_TML(y, pred, -pred)
                
                # ONLY MSE loss is used!
                loss =  loss_fn_MSE(pred,y)

                # Proposed loss is used!
                # loss = 3 * loss_fn_MSE(pred, y) + one_hot_loss_weight * loss_fn_CE(pred_softmax ,one_hot_label_return(y).squeeze()) \
                # + loss_fn_MSE(torch.square(pred).sum(dim = 1).to(args.device), torch.ones_like(pred.sum(dim = 1)).to(args.device)) + loss_fn_TML(y, pred, -pred)

                # loss_fn_TML(y, pred, -pred) --> 위에꺼 되면 추가하기
                # loss = 40 * loss_fn_MSE(pred,y) + one_hot_loss_weight * loss_fn_CE(pred_softmax ,one_hot_label_return(y).squeeze()) \
                # + 0.5 * loss_fn_MRL(pred, -base_constraint, torch.tensor([+1]).to(args.device)) + 0.5 * loss_fn_MRL(pred, base_constraint, torch.tensor([-1]).to(args.device))
                # loss = loss_fn_MSE(pred,y) + loss_fn_CE(pred_softmax ,one_hot_label_return(y).squeeze()) \
                # + loss_fn_MRL(pred_softmax, base_constraint ,torch.tensor([+1]).to(args.device))

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Scheduler should be applied following each epoch
            scheduler.step()
            #print("LR: {}".format(optimizer.param_groups[0]['lr']))

            epoch_loss = epoch_loss / (batch+1)

            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_train_model_PATH)
                print("\nBest one so far: [Epoch {:>2}] [Loss {:.4f}]\n".format(best_epoch, 2 * best_epoch_loss))

            print("[Epoch {:>2}] Average loss per epoch = {:.4f}".format(epoch+1, 2 * epoch_loss))
        
        print("\nBest train loss: [Epoch {:>2}] [Loss {:.4f}]\n".format(best_epoch, 2 * best_epoch_loss))

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


    elif args.filter_type == 'LS':
        input_file_name = 'filter_input_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(test_symb_len, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_train)
        input_file_PATH = './data/symbol_tensor/train_data/' + input_file_name + '.npy'

        target_file_name = 'filter_target_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(test_symb_len // L, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_train)
        target_file_PATH = './data/symbol_tensor/train_data/' + target_file_name + '.npy'    

        if os.path.isfile(target_file_PATH) and os.path.isfile(input_file_PATH):
            RX_test_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]
            target_symb = np.load(target_file_PATH)[:,0] + 1j * np.load(target_file_PATH)[:,1]
        
        p_hat = np.zeros((args.filter_size, 1), dtype = np.complex_)
        R_hat = np.zeros((args.filter_size, args.filter_size), dtype = np.complex_)

        for set_idx in range(RX_test_symb.shape[0]):
            p_hat += np.conj(target_symb[set_idx]) * RX_test_symb[set_idx].reshape(-1,1)
            R_hat += R_calc(RX_test_symb[set_idx])

        p_hat /= RX_test_symb.shape[0]
        R_hat /= RX_test_symb.shape[0] 

        w_LS = np.linalg.inv(R_hat) @ p_hat
        
        # print("p: {}, R: {}, w: {}".format(p_hat, R_hat, w_LS))
        

    ################
    # Testing part # 
    # LMMSE is implemented only here
    ################
    
    if args.filter_type == 'NN':
        model = NF(args.total_taps, args.filter_size).to(args.device)
        model.load_state_dict(torch.load(best_train_model_PATH))
        model.eval()

        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for batch, (X,y) in enumerate(test_dataloader):
                X, y = X.to(args.device), y.unsqueeze(2).to(args.device)
                pred = model(X)
                loss = loss_fn_MSE(pred,y)
                # import ipdb; ipdb.set_trace()
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
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.4f} (%)], [Epochs {}], [Batch size {}], [Filter size {}], [LR {}], [Scheduler type {}], [Scheduler gamma {}], [Decision delay {}], [Total taps {}], [SNR {} (dB)], [Train/test seq length {}/{}], [Random seed (Channel) {}], [Random seed (Train) {}], [Random seed (Test) {}], [Date {}]"\
            .format(args.filter_type, 2*test_loss, 100*SER, args.epochs, args.bs, args.filter_size, args.lr, scheduler_type, scheduler_gamma, args.decision_delay, args.total_taps, args.SNR, args.train_seq_len, args.test_seq_len, args.rand_seed_channel, args.rand_seed_train, args.rand_seed_test, time.ctime()))
            
        with open('./results/channel_MSE.txt','a') as f:
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.4f} (%)], [Random seed (Channel) {}], [Channel {}], [Filter size {}], [Decision delay {}], [Total taps {}], [Date {}]"\
            .format(args.filter_type, 2*test_loss, 100*SER, args.rand_seed_channel, channel_taps.T[0], args.filter_size, args.decision_delay, args.total_taps, time.ctime()))

    else:
        input_file_name = 'filter_input_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(test_symb_len, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_test)
        input_file_PATH = './data/symbol_tensor/test_data/' + input_file_name + '.npy'

        if args.filter_type == 'Linear':
            target_file_name = 'symb_len_{}_mod_{}_S_{}'.format(test_symb_len, args.mod_scheme, args.rand_seed_test)
        elif args.filter_type == 'LMMSE' or args.filter_type == 'LS':
            target_file_name = 'filter_target_len_{}_filter_size_{}_mod_{}_D_{}_S_{}'.format(test_symb_len // L, args.filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_test)

        target_file_PATH = './data/symbol_tensor/test_data/' + target_file_name + '.npy'        

        if os.path.isfile(target_file_PATH):
            target_symb = np.load(target_file_PATH)[:,0] + 1j * np.load(target_file_PATH)[:,1]

        if os.path.isfile(input_file_PATH) and args.filter_type == 'Linear':
            RX_test_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]
            total_symb_num = RX_test_symb.shape[0]
            opt_test_MSE = np.square(np.abs(np.matmul(RX_test_symb, LF_weight).T - target_symb.reshape(1,-1))).mean()
        
        elif os.path.isfile(input_file_PATH) and (args.filter_type == 'LMMSE' or args.filter_type == 'LS'):
            # Solve by matrix calculation
            RX_test_symb = np.load(input_file_PATH)[:,0] + 1j * np.load(input_file_PATH)[:,1]
            if args.filter_type == 'LMMSE':
                SNR_ratio = 10**(args.SNR / 10)
                channel_col = channel_matrix[:,args.decision_delay].reshape(-1,1)
                
                # LMMSE weight vector // C @ np.conj(C).T = Hermitian & np.linalg.inv(C) = inverse matrix
                # w_LMMSE.shape is (filter_size, 1)
                # w_LMMSE = np.linalg.inv(channel_matrix @ np.conj(channel_matrix).T) @ channel_col
                
                w_LMMSE = np.linalg.inv(channel_matrix @ np.conj(channel_matrix).T + 1/SNR_ratio * np.eye(args.filter_size)) @ channel_col
                w_linear = w_LMMSE

            elif args.filter_type == 'LS':
                w_linear = w_LS
    
            opt_test_MSE = 0

            correct = 0
            
            for set_idx in range(RX_test_symb.shape[0]):
                pred_symb = np.conj(w_linear).T @ RX_test_symb[set_idx].reshape(-1,1)
                if (np.real(target_symb[set_idx]) * np.real(pred_symb) > 0) and (np.imag(target_symb[set_idx]) * np.imag(pred_symb) > 0):
                    correct += 1
                opt_test_MSE += np.square(np.abs(target_symb[set_idx] - pred_symb)).mean()

            SER = (RX_test_symb.shape[0] - correct) / RX_test_symb.shape[0] 
            print("\nSymbol error rate (SER): {:.2f} %".format(100 * SER))
            
            opt_test_MSE /= RX_test_symb.shape[0]

        # MSE for a single symbol
        print("\nOptimal test MSE value (per single symbol): {:.4f}".format(opt_test_MSE))

        with open('./results/MSE_test_results.txt','a') as f:
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.4f} (%)], [Filter size {}], [Decision delay {}], [Total taps {}], [SNR {} (dB)], [Train/test seq length {}/{}], [Random seed (Channel) {}], [Random seed (Train) {}], [Random seed (Test) {}], [Date {}]"\
            .format(args.filter_type, opt_test_MSE, 100 * SER, args.filter_size, args.decision_delay, args.total_taps, args.SNR, args.train_seq_len, args.test_seq_len, args.rand_seed_channel, args.rand_seed_train, args.rand_seed_test, time.ctime()))
            
        with open('./results/channel_MSE.txt','a') as f:
            f.write("\n[Filter type {}], [MSE {:.4f}], [SER {:.4f} (%)], [Random seed (Channel) {}], [Channel {}], [Filter size {}], [Decision delay {}], [Total taps {}], [Date {}]"\
            .format(args.filter_type, opt_test_MSE, 100 * SER, args.rand_seed_channel, channel_taps.T[0], args.filter_size, args.decision_delay, args.total_taps, time.ctime()))
        
    # tensorboard --logdir=runs
    # tensorboard --inspect --event_file=myevents.out --tag=loss
    print("\n{} filter simulation is DONE!\n".format(args.filter_type))
