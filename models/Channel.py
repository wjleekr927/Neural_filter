import numpy as np 
import sys 
import os
import os.path
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
    
    # Save the channel vector
    channel_tap_name = 'channel_{}_taps_S_{}'.format(total_taps, seed)
    channel_tap_PATH = './models/channel_tap_vector/' + channel_tap_name + '.npy'

    if not os.path.isfile(channel_tap_PATH):
        np.save(channel_tap_PATH, channel_vec)

    print("\n{} channel taps with seed {} are used".format(total_taps, seed))

    return channel_vec

# symbol_tensor shape => (n,)
def apply_channel(channel_taps, filter_size, filter_type, train_symbol_tensor, test_symbol_tensor):
    args = args_parser()

    L = len(channel_taps) + filter_size - 1
    
    # Shape of 'symbol_tensor' is (n,) with complex form 
    train_symbol_tensor, train_symbol_num = train_symbol_tensor.reshape(-1,1), train_symbol_tensor.shape[-1]
    test_symbol_tensor, test_symbol_num  = test_symbol_tensor.reshape(-1,1), test_symbol_tensor.shape[-1]
    
    channel_matrix = np.zeros((filter_size, L), dtype = 'complex_')

    for idx in range(len(channel_taps)):
        channel_matrix += np.eye(filter_size, L, k=idx) * channel_taps[idx]

    train_applied, test_applied = [], []

    # Apply to every L symbols
    # Applied variable shape: (Number of set, filter_size, 1)
    for set_idx in range(train_symbol_num // L):
        train_applied.append(channel_matrix @ train_symbol_tensor[set_idx*L : (set_idx+1)*L])

    for set_idx in range(test_symbol_num // L):
        test_applied.append(channel_matrix @ test_symbol_tensor[set_idx*L : (set_idx+1)*L])

    # Implement filter train / test data
    if args.filter_type == 'NN' or args.filter_type == 'Linear':
        train_filter_input_np, test_filter_input_np = np.asarray(train_applied).squeeze(-1), np.asarray(test_applied).squeeze(-1)

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
        # Data shape? (set of data, 2, filter_size)
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