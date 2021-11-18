# Take arguments
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # Model setting arguments
    parser.add_argument('--seq_len', type=int, default=40000, help="Length of data sequence")
    parser.add_argument('--filter_type', type=str, default='NN', help = "Filter type: NN or Linear")
    parser.add_argument('--filter_size', type=int, default=18, help = "Size of filter")
    parser.add_argument('--total_taps', type=int, default= 18, help = "Number of channel taps")
    parser.add_argument('--mod_scheme', type= str, default= 'QPSK', help="Modulation scheme")
    # parser.add_argument('--train_ratio', type= float, default= .7, help="Ratio of training set (0<=R<=1)")

    parser.add_argument('--rand_seed', type= int, default= 9999, help="Random seed setting")
    parser.add_argument('--data_gen_type', type= str, default= 'train', help="Type of data to be generated: 'train' or 'test'")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index setting, -1 for CPU")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs")
    parser.add_argument('--bs', type=int, default=8, help="Size of batch")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

    # parser.add_argument('--output_dir', type=str, default='./output')
    # parser.add_argument('--summary_grad', action='store_true')
    # parser.add_argument('--noise', action = 'store_true', help= "Add option to use noise. Defalut is not used. Also, power needed to be selected in code.")

    args = parser.parse_args()
    return args