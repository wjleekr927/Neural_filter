# Import packages
import sys
import torch
import numpy as np

from utils.options import args_parser

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm
from tqdm import trange

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Parse args (from ./utils/options.py)
    args = args_parser()

    # Device setting
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Print for current option
    print("\nTraining for following settings:")
    print("\tEpoch: {}, Input M: {}, Model: {}, Device: {}".format(args.epochs, args.M_input, args.model, args.device))
    
    # tensorboard --logdir=runs
    # tensorboard --inspect --event_file=myevents.out --tag=loss
    print("\nDone!\n")