# Take arguments
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # Model setting arguments, following should be satisfied L <= M
    parser.add_argument('--gen_seq_len', type=int, default=40000, help="Length of data sequence to be generated")
    parser.add_argument('--train_seq_len', type=int, default=40000, help="Length of train data sequence")
    parser.add_argument('--test_seq_len', type=int, default=40000, help="Length of test data sequence")
    parser.add_argument('--filter_type', type=str, default='NN', help = "Filter type: NN or Linear or LMMSE")
    # (18,18) was default, changed to (24,10) 
    parser.add_argument('--filter_size', type=int, default=24, help = "Size of filter")
    parser.add_argument('--total_taps', type=int, default= 10, help = "Number of channel taps")
    parser.add_argument('--SNR', type=float, default= 10, help = "Signal to noise ratio (dB scale)")
    parser.add_argument('--decision_delay', type=int, default= 0, help = "Decision delay tau")
    parser.add_argument('--decay_factor', type=float, default= 0, help="Exponential tap decay factor k: exp(-k)")
    parser.add_argument('--mod_scheme', type= str, default= 'QPSK', help="Modulation scheme")
    parser.add_argument('--scatter_plot', action= 'store_true', help="Use this option if you want to plot the constellation scatter. Default: not draw")
    # parser.add_argument('--train_ratio', type= float, default= .7, help="Ratio of training set (0<=R<=1)")

    parser.add_argument('--rand_seed_train', type= int, default= 9999, help="Random seed setting for training set")
    parser.add_argument('--rand_seed_test', type= int, default= 4999, help="Random seed setting for test set")
    parser.add_argument('--rand_seed_channel', type= int, default= 2077, help="Random seed setting for channel taps")
    parser.add_argument('--data_gen_type', type= str, default= 'train', help="Type of data to be generated: 'train' or 'test'")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index setting, -1 for CPU")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=80, help="Number of epochs")
    parser.add_argument('--bs', type=int, default=256, help="Size of batch")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

    # parser.add_argument('--output_dir', type=str, default='./output')
    # parser.add_argument('--summary_grad', action='store_true')
    # parser.add_argument('--noise', action = 'store_true', help= "Add option to use noise. Defalut is not used. Also, power needed to be selected in code.")

    args = parser.parse_args()
    return args