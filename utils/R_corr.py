import numpy as np

def R_calc(symb_seq, RX_num):
    R_hat = np.zeros((len(symb_seq), len(symb_seq)), dtype = np.complex_)
    for idx in range(len(symb_seq) // RX_num):
        R_hat_elem = R_elem_calc(symb_seq, idx, RX_num)
        if idx != 0:
            R_hat += np.kron(np.eye(len(symb_seq) // RX_num , k = idx), R_hat_elem)
            # 안 되면, 아래가 유력 후보, conj transpose가 아니라 conj만!
            R_hat += np.kron(np.eye(len(symb_seq) // RX_num , k = -idx), np.conj(R_hat_elem).T)
        elif idx == 0:
            R_hat += np.kron(np.eye(len(symb_seq) // RX_num , k = idx), R_hat_elem)
    return R_hat

def R_elem_calc(symb_seq, tau, RX_num):
    symb_seq = np.flip(symb_seq)
    R_elem = np.zeros((RX_num, RX_num), dtype = np.complex_)
    norm_const = len(symb_seq) // RX_num - tau
    for symb_idx in range(norm_const):
        # np.flip is applied
        R_elem += np.flip(symb_seq[RX_num*(symb_idx + tau):RX_num*(symb_idx + tau + 1)].reshape(-1,1)) @ \
            np.flip(np.conj(symb_seq[RX_num*symb_idx:RX_num*(symb_idx+1)].reshape(-1,1)).T)

    R_elem /= norm_const

    return R_elem