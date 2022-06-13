import numpy as np

def R_calc(symb_seq):
    R_hat = np.zeros((len(symb_seq), len(symb_seq)), dtype = np.complex_)
    #import ipdb; ipdb.set_trace()
    for idx in range(len(symb_seq)):
        R_hat_elem = R_elem_calc(symb_seq, idx)
        if idx != 0:
            R_hat += R_hat_elem * np.eye(len(symb_seq), k = idx)
            R_hat += np.conj(R_hat_elem) * np.eye(len(symb_seq), k = -idx)
        elif idx == 0:
            R_hat += R_hat_elem * np.eye(len(symb_seq), k = idx)
    return R_hat

def R_elem_calc(symb_seq, tau):
    symb_seq = np.flip(symb_seq)
    R_elem = 0
    norm_const = len(symb_seq) - tau
    for symb_idx in range(norm_const):
        R_elem += symb_seq[symb_idx + tau] * np.conj(symb_seq[symb_idx])

    R_elem /= norm_const

    return R_elem