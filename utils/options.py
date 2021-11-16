# Take arguments
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument('--epochs', type=int, default=5 , help="Number of epochs")
    parser.add_argument('--bs', type=int, default=16, help="Size of batch")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

    parser.add_argument('--M_input', type=int, default=10, help="Number of input sensors")
    parser.add_argument('--train_set_ratio', type= float, default= .7, help="Ratio of training set")
    # parser.add_argument('--optimizer', type=str, default='SGD')
    # parser.add_argument('--output_dir', type=str, default='./output')
    # parser.add_argument('--summary_grad', action='store_true')

    # Model arguments
    parser.add_argument('--model', type=str, default='deep',help = "Model structure setting")

    # Others
    # parser.add_argument('--noise', action = 'store_true', help= "Add option to use noise. Defalut is not used. Also, power needed to be selected in code.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index setting, -1 for CPU")
    args = parser.parse_args()
    return args
