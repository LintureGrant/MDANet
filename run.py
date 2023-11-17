import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj','human','KTH'])
    parser.add_argument('--num_workers', default=0, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    parser.add_argument('--out_shape', default=[10, 1, 64, 64], type=int, nargs='*')
    parser.add_argument('--kernel_size', default=[3, 3, 3, 3], type=int, nargs='*')
    parser.add_argument('--hid_channel', default=64, type=int)
    parser.add_argument('--layer_num', default=4, type=int)
    parser.add_argument('--reduction', default=2, type=int)
    parser.add_argument('--layer_config', default=(1, 8, 2, 8), type=int)
    parser.add_argument('--group_param', default=(2, 4), type=int)

    # Training parameters
    parser.add_argument('--epochs', default=2001, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)