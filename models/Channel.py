import numpy as np 
import sys 
sys.path.insert(1,'..')
from utils.options import args_parser

def channel_gen(total_taps, decay_factor, seed):
    np.random.seed(seed)
    channel_vec = np.zeros((total_taps,1), dtype = 'complex_')
    # Not use exponential here
    # exp_decay = np.round(np.exp(-decay_factor), 4)
    for idx in range(total_taps):
        # Std to variance
        rnd_complex = np.random.normal(scale=1/(2*total_taps)**.5) + np.random.normal(scale=1/(2*total_taps)**.5) * 1j
        channel_vec[idx] = rnd_complex
    
    return channel_vec

# symbol_tensor shape => (n,)
def apply_channel(channel_taps, filter_size, filter_type, train_symbol_tensor, test_symbol_tensor):
    args = args_parser()
    
    train_symbol_tensor, train_symbol_num = train_symbol_tensor.reshape(1,-1), train_symbol_tensor.shape[-1]
    test_symbol_tensor, test_symbol_num  = test_symbol_tensor.reshape(1,-1), test_symbol_tensor.shape[-1]
    
    channel_for_train = np.zeros((train_symbol_num,train_symbol_num), dtype = 'complex_')
    channel_for_test = np.zeros((test_symbol_num,test_symbol_num), dtype = 'complex_')

    for idx in range(len(channel_taps)):
        channel_for_train += np.eye(train_symbol_num, k=idx) * channel_taps[idx]
        channel_for_test += np.eye(test_symbol_num, k=idx) * channel_taps[idx]

    # Shape: (n,1)
    train_applied = (train_symbol_tensor @ channel_for_train).T
    test_applied = (test_symbol_tensor @ channel_for_test).T

    if args.filter_type == 'NN' or args.filter_type == 'Linear':
        train_filter_input_list, test_filter_input_list = [], []
        train_filter_window, test_filter_window  = [0] * filter_size, [0] * filter_size
        
        # Implement filter train data
        for idx in range(train_symbol_num):
            train_filter_window = train_filter_window[1:]
            train_filter_window.append(train_applied[idx].item())
            train_filter_input_list.append(train_filter_window)
        
        # Implement filter test data
        for idx in range(test_symbol_num):
            test_filter_window = test_filter_window[1:]
            test_filter_window.append(test_applied[idx].item())
            test_filter_input_list.append(test_filter_window)

        train_filter_input_np, test_filter_input_np = np.array(train_filter_input_list), np.array(test_filter_input_list)

        # To make two channels, expand the dimensions
        train_filter_input_IQ_np = np.concatenate((np.expand_dims(np.real(train_filter_input_np),axis=1),\
        np.expand_dims(np.imag(train_filter_input_np), axis=1)),axis = 1)
        
        test_filter_input_IQ_np = np.concatenate((np.expand_dims(np.real(test_filter_input_np),axis=1),\
        np.expand_dims(np.imag(test_filter_input_np), axis=1)),axis = 1)

        train_data_name = "./data/symbol_tensor/train_data" + "/filter_input_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(str(train_symbol_num), filter_size, args.mod_scheme, args.rand_seed_train)

        test_data_name = "./data/symbol_tensor/test_data" + "/filter_input_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(str(test_symbol_num), filter_size, args.mod_scheme, args.rand_seed_test)

        # Save to numpy
        np.save(train_data_name, train_filter_input_IQ_np)
        np.save(test_data_name, test_filter_input_IQ_np)
    
    # Doesn't need to be divided into two channels (real, imag)
    elif args.filter_type == 'Optimal_Linear':
        train_filter_input_np = train_applied.reshape(int(args.filter_size**(0.5)), -1, order='F')
        test_filter_input_np = test_applied.reshape(int(args.filter_size**(0.5)), -1, order='F')
        
        train_data_name = "./data/symbol_tensor/train_data" + "/opt_filter_input_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(str(train_symbol_num), filter_size, args.mod_scheme, args.rand_seed_train)

        test_data_name = "./data/symbol_tensor/test_data" + "/opt_filter_input_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(str(test_symbol_num), filter_size, args.mod_scheme, args.rand_seed_test)
        
        # Save to numpy
        np.save(train_data_name, train_filter_input_np)
        np.save(test_data_name, test_filter_input_np)

    return None