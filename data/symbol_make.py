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

    # L = Nc + M - 1, Nc: channel taps / M: filter size
    L = args.filter_size + args.total_taps - 1

    # N_train: length of original sequence, implemented by 'args.gen_seq_len' (which means total number of bits)
    # L symbols (x) are used to make M symbols (y)

    if args.data_gen_type == 'train':
        np.random.seed(args.rand_seed_train)
    elif args.data_gen_type == 'test':
        np.random.seed(args.rand_seed_test)
    else:
        raise Exception("Data generation type should be 'train' or 'test'")

    rand_seq = np.round(np.random.random_sample(args.gen_seq_len)).astype(int)

    symb_list = []

    if args.mod_scheme == 'QPSK':
        # Normalized to power = 1
        norm_cof = np.round(1 / np.sqrt(2), 4)
        symb_dic = {'00': (norm_cof + norm_cof*1j), '10': (norm_cof - norm_cof*1j), \
        '11': (-norm_cof - norm_cof*1j), '01': (-norm_cof + norm_cof*1j)}

        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        symb_len = args.gen_seq_len // bits_per_symb

        if symb_len % L != 0:
            # symb_len / L: number of data group to be used
            raise Exception("Total length of sequence should be divided by L")

    # Append a corresponding symbol to the list
    for idx in range(symb_len):
        ith_symb = symb_dic[str(rand_seq[2*idx]) + str(rand_seq[2*idx + 1])]
        symb_list.append(ith_symb)

    symb_np = np.array(symb_list).reshape(-1,1)
    symb_IQ_np = np.concatenate((np.real(symb_np), np.imag(symb_np)), axis = 1)
    
    if args.data_gen_type == 'train':
        data_name = "./symbol_tensor/train_data" + "/symb_len_{}_mod_{}_S_{}".format(str(symb_len), args.mod_scheme, args.rand_seed_train)
        output_file_name = "./symbol_tensor/train_data" + "/filter_target_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(symb_len // L, args.filter_size, args.mod_scheme, args.rand_seed_train)
    else:
        data_name = "./symbol_tensor/test_data" + "/symb_len_{}_mod_{}_S_{}".format(str(symb_len), args.mod_scheme, args.rand_seed_test)
        output_file_name = "./symbol_tensor/test_data" + "/filter_target_len_{}_filter_size_{}_mod_{}_S_{}"\
        .format(symb_len // L, args.filter_size, args.mod_scheme, args.rand_seed_test)
    
    # symb_IQ_np.shape => (number of symbol, 2)
    # Maybe Dataloader handle it automatically

    # Stride: L
    # (04/09) x[0]를 x[L]처럼 처리하는 것 같아서 일단 얘만 바꿔보겠습니다 
    # => 시도해봤는데 안되네요
    # symb_IQ_np_sampled = symb_IQ_np[L-1:symb_len:L][:]
    symb_IQ_np_sampled = symb_IQ_np[::L][:]

    np.save(data_name, symb_IQ_np)
    np.save(output_file_name, symb_IQ_np_sampled)
    
    print("Success the generation of {} symbol tensor data!".format(args.data_gen_type))
    # print(">> Format example: 1-1j to [1,-1], Shape: (# of symbol, 2)")