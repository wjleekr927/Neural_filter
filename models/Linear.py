# LF
import cvxpy as cp
import torch
from torch import nn

# Base class definition
class LF:
    def __init__(self, RX_symb, target_symb, filter_size):
        self.RX_symb = RX_symb
        self.target_symb = target_symb
        self.total_symb_num = RX_symb.shape[0]
        self.filter_size = filter_size

    def optimize(self):
        LF_weight = cp.Variable((self.filter_size, 1), complex = True)
        objective = cp.Minimize(cp.sum_squares((self.RX_symb @ LF_weight).T - self.target_symb.reshape(1,-1)))
        prob = cp.Problem(objective)
        rst = prob.solve()
        # LF_weight is for test phase
        return LF_weight.value, rst