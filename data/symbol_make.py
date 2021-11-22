import numpy as np
import torch

from tqdm import tqdm
from tqdm import trange

from queue import Queue

import sys 
sys.path.insert(1,'..')

from utils.options import args_parser

if __name__ == '__main__':
    # From options.py
    args = args_parser()

    if args.data_gen_type == 'train':
        np.random.seed(args.rand_seed_train)
    elif args.data_gen_type == 'test':
        np.random.seed(args.rand_seed_test)
    else:
        raise Exception("Data generation type should be 'train' or 'test'")

    rand_seq = np.round(np.random.random_sample(args.seq_len)).astype(int)

    symb_list = []

    if args.mod_scheme == 'QPSK':
        # Normalized to power = 1
        norm_cof = np.round(1 / np.sqrt(2), 4)
        symb_dic = {'00': (norm_cof + norm_cof*1j), '10': (norm_cof - norm_cof*1j), \
        '11': (-norm_cof - norm_cof*1j), '01': (-norm_cof + norm_cof*1j)}

        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        symb_len = args.seq_len // bits_per_symb

    # Append a corresponding symbol to the list
    for idx in range(symb_len):
        ith_symb = symb_dic[str(rand_seq[2*idx]) + str(rand_seq[2*idx + 1])]
        symb_list.append(ith_symb)

    symb_np = np.array(symb_list).reshape(-1,1)
    symb_IQ_np = np.concatenate((np.real(symb_np), np.imag(symb_np)),axis = 1)
    
    if args.data_gen_type == 'train':
        data_name = "./symbol_tensor/train_data" + "/symb_len_{}_mod_{}_S_{}".format(str(symb_len), args.mod_scheme, args.rand_seed_train)
    else:
        data_name = "./symbol_tensor/test_data" + "/symb_len_{}_mod_{}_S_{}".format(str(symb_len), args.mod_scheme, args.rand_seed_test)
    np.save(data_name, symb_IQ_np)

    filter_input_list = []
    filter_window = [0] * args.filter_size
    
    # Implement filter
    for idx in range(symb_len):
        filter_window = filter_window[1:]
        filter_window.append(symb_np[idx].item())
        filter_input_list.append(filter_window)

    filter_input_np = np.array(filter_input_list)
    # To make two channels, expand the dimensions
    filter_input_IQ_np = np.concatenate((np.expand_dims(np.real(filter_input_np),axis=1),\
    np.expand_dims(np.imag(filter_input_np), axis=1)),axis = 1)

    if args.data_gen_type == 'train':
        data_name = "./symbol_tensor/train_data" + "/filter_input_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(str(symb_len), args.filter_size, args.mod_scheme, args.rand_seed_train)
    else:
        data_name = "./symbol_tensor/test_data" + "/filter_input_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(str(symb_len), args.filter_size, args.mod_scheme, args.rand_seed_test)
    np.save(data_name, filter_input_IQ_np)
    
    print("Symbol tensor {} data generation success!".format(args.data_gen_type))