# Viterbi algorithm
import numpy as np 
import itertools
import sys 
import os
import os.path
sys.path.insert(1,'..')
from utils.options import args_parser

# received_y: M x 1
def Viterbi_decoding(received_y, channel_vec, decision_delay, scheme):
    # Index becomes to be reversed
    received_y = np.flip(received_y)
    channel_taps = len(channel_vec)
    L = len(received_y) + channel_taps - 1
    
    if scheme == 'QPSK':
        # 2 bits for one symbol in QPSK
        bits_per_symb = 2
        norm_cof = np.round(1 / np.sqrt(2), 4)
        symb_GT_list = [[[+norm_cof], [+norm_cof]], [[+norm_cof], [-norm_cof]], [[-norm_cof], [+norm_cof]], [[-norm_cof], [-norm_cof]]]
    elif scheme == '16QAM':
        # 4 bits for one symbol in 16QAM
        bits_per_symb = 4
        train_symb_len = args.train_seq_len // bits_per_symb
        test_symb_len = args.test_seq_len // bits_per_symb
        
        # Assume same Eb energy
        norm_cof = np.round(1 / np.sqrt(5), 4)
        QAM16_label_GT_list = [[[+norm_cof], [+norm_cof]], [[+3*norm_cof], [+norm_cof]], [[+norm_cof], [+3*norm_cof]], [[+3*norm_cof], [+3*norm_cof]], \
            [[+norm_cof], [-norm_cof]], [[+3*norm_cof], [-norm_cof]], [[+norm_cof], [-3*norm_cof]], [[+3*norm_cof], [-3*norm_cof]], \
                [[-norm_cof], [+norm_cof]], [[-3*norm_cof], [+norm_cof]], [[-norm_cof], [+3*norm_cof]], [[-3*norm_cof], [+3*norm_cof]], \
                    [[-norm_cof], [-norm_cof]], [[-3*norm_cof], [-norm_cof]], [[-norm_cof], [-3*norm_cof]], [[-3*norm_cof], [-3*norm_cof]]]
    
    total_symb_types = len(symb_GT_list)
    all_permutation_idx = list(itertools.product(range(total_symb_types), repeat = channel_taps))
    full_cost_dict, sub_cost_dict = {}, {}
    determined_x_idx_list = []
    
    y_idx = 0

    while y_idx < len(received_y):
        target_y = received_y[y_idx]
        if y_idx == 0:
            for idx, elem in enumerate(all_permutation_idx):
                ref_x = np.squeeze(np.array([symb_GT_list[i] for i in elem]))
                ref_x_IQ = (ref_x[:,0] + ref_x[:,1] * 1j).reshape(-1,1)
                
                pred_y = channel_vec.reshape(1,-1) @ np.flip(ref_x_IQ)
                full_cost_dict[elem] = float(abs(target_y - pred_y))

        else:
             for idx, elem in enumerate(sub_permutation_idx):
                for symb_idx in range(total_symb_types):
                    full_idx = elem + (symb_idx, )
                    ref_x = np.squeeze(np.array([symb_GT_list[i] for i in full_idx]))
                    ref_x_IQ = (ref_x[:,0] + ref_x[:,1] * 1j).reshape(-1,1)
                    
                    pred_y = channel_vec.reshape(1,-1) @ np.flip(ref_x_IQ)
                    full_cost_dict[full_idx] = sub_cost_dict[elem] + float(abs(target_y - pred_y))
                    
        # import ipdb; ipdb.set_trace()                 
        min_cost_idx_tuple = min(full_cost_dict, key = full_cost_dict.get)
        
        if y_idx != (len(received_y) - 1):
            determined_x_idx_list.append(min_cost_idx_tuple[0])
        else:
            determined_x_idx_list = determined_x_idx_list + list(min_cost_idx_tuple)
            break
            
        # To check idx permutation except first element
        sub_permutation_idx = list(itertools.product(range(total_symb_types), repeat = channel_taps - 1))
        
        for elem in sub_permutation_idx:
            # Set to sufficiently large number (as 99 here)
            min_val = 99
            for symb_idx in range(total_symb_types):
                full_idx = (symb_idx, ) + elem
                if min_val > full_cost_dict[full_idx]:
                    min_val = full_cost_dict[full_idx]
            sub_cost_dict[elem] = min_val 
            
        y_idx += 1
        
    pred_symb_x_idx = determined_x_idx_list[L - decision_delay - 1]
    return np.squeeze(symb_GT_list[pred_symb_x_idx])[0] + np.squeeze(symb_GT_list[pred_symb_x_idx])[1] * 1j