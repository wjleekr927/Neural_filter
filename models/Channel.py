import numpy as np 
import sys 
import os
import os.path
sys.path.insert(1,'..')
from utils.options import args_parser

def channel_gen(total_taps, decay_factor, RX_num, seed):
    np.random.seed(seed)
    channel_vec = np.zeros((total_taps, RX_num), dtype = 'complex_')
    # import ipdb; ipdb.set_trace()
    # Not use exponential here
    exp_decay = np.round(np.exp(-decay_factor), 4)
    for row_idx in range(total_taps):
        for col_idx in range(RX_num):
            # Std to variance
            rnd_complex = exp_decay**row_idx * (np.random.normal(scale=1/(2*total_taps)**.5) + np.random.normal(scale=1/(2*total_taps)**.5) * 1j)
            channel_vec[row_idx, col_idx] = rnd_complex

    # Save the channel vector
    channel_tap_vector_name = 'channel_{}_taps_RX_{}_S_{}'.format(total_taps, RX_num, seed)
    channel_tap_vector_PATH = './models/channel_tap_vector/' + channel_tap_vector_name + '.npy'

    # If you don't want to overwrite the file, activate the below line
    # if not os.path.isfile(channel_tap_vector_PATH):
    np.save(channel_tap_vector_PATH, channel_vec)

    print("\n{} taps channel with {} receiver(s) and seed {} are used".format(total_taps, RX_num, seed))

    return channel_vec

# symbol_tensor shape => (n,)
def apply_channel(channel_taps, filter_size, filter_type, train_symbol_tensor, test_symbol_tensor, RX_num, seed):
    args = args_parser()

    L = len(channel_taps) + filter_size - 1
    
    # Shape of 'symbol_tensor' is (n,) with complex form 
    train_symbol_tensor, train_symbol_num = train_symbol_tensor.reshape(-1,1), train_symbol_tensor.shape[-1]
    test_symbol_tensor, test_symbol_num  = test_symbol_tensor.reshape(-1,1), test_symbol_tensor.shape[-1]
    
    channel_matrix = np.zeros((RX_num * filter_size, L), dtype = 'complex_')

    for idx in range(len(channel_taps)):
        # Before SIMO (=before using kron)
        # channel_matrix += np.eye(filter_size, L, k=idx) * channel_taps[idx]
        channel_matrix += np.kron(np.eye(filter_size, L, k=idx), channel_taps[idx].reshape(-1,1))
        

    # Save the channel matrix
    channel_tap_matrix_name = 'channel_matrix_{}_taps_RX_{}_filter_size_{}_S_{}'.format(len(channel_taps), RX_num, filter_size, seed)
    channel_tap_matrix_PATH = './models/channel_tap_matrix/' + channel_tap_matrix_name + '.npy'

    # If you don't want to overwrite the file, activate the below line
    # if not os.path.isfile(channel_tap_matrix_PATH) or 1:
    np.save(channel_tap_matrix_PATH, channel_matrix)
    
    train_applied, test_applied = [], []

    # Recall the symbol energy is 1
    # Then, complex noise variance = 1/SNR 
    SNR_ratio = 10**(args.SNR / 10)
    noise_var = 1/SNR_ratio

    # Apply to every L symbols
    # Applied variable shape: (Number of set, filter_size, 1)

    for set_idx in range(train_symbol_num // L):
        noise_vec = np.random.normal(0, np.sqrt(noise_var / 2), size = (RX_num * filter_size, 1)) \
        + np.random.normal(0, np.sqrt(noise_var / 2), size = (RX_num * filter_size, 1)) * 1j
        train_applied.append(channel_matrix @ np.flip(train_symbol_tensor[set_idx*L : (set_idx+1)*L]) + noise_vec)

    for set_idx in range(test_symbol_num // L):
        noise_vec = np.random.normal(0, np.sqrt(noise_var / 2), size = (RX_num * filter_size, 1)) \
        + np.random.normal(0, np.sqrt(noise_var / 2), size = (RX_num * filter_size, 1)) * 1j
        test_applied.append(channel_matrix @ np.flip(test_symbol_tensor[set_idx*L : (set_idx+1)*L]) + noise_vec)

    # Implement filter train / test data
    if args.filter_type == 'NN' or args.filter_type == 'Linear' or args.filter_type == 'LMMSE' or args.filter_type == 'LS' or args.filter_type == 'Viterbi':
        train_filter_input_np, test_filter_input_np = np.asarray(train_applied).squeeze(-1), np.asarray(test_applied).squeeze(-1)

        # To make two channels, expand the dimensions
        train_filter_input_IQ_np = np.concatenate((np.expand_dims(np.real(train_filter_input_np),axis=1),\
        np.expand_dims(np.imag(train_filter_input_np), axis=1)),axis = 1)
        
        test_filter_input_IQ_np = np.concatenate((np.expand_dims(np.real(test_filter_input_np),axis=1),\
        np.expand_dims(np.imag(test_filter_input_np), axis=1)),axis = 1)

        train_data_name = "./data/symbol_tensor/train_data" + "/filter_input_len_{}_RX_{}_filter_size_{}_mod_{}_D_{}_S_{}"\
        .format(str(train_symbol_num), RX_num, filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_train)

        test_data_name = "./data/symbol_tensor/test_data" + "/filter_input_len_{}_RX_{}_filter_size_{}_mod_{}_D_{}_S_{}"\
        .format(str(test_symbol_num), RX_num, filter_size, args.mod_scheme, args.decision_delay, args.rand_seed_test)

        # Save as numpy
        # Data shape? (set of data, 2, filter_size)
        np.save(train_data_name, train_filter_input_IQ_np)
        np.save(test_data_name, test_filter_input_IQ_np)
    
    return None