import os
import argparse

from torch.backends import cudnn
from utils.utils import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.utils import *

from solver import Solver
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):

        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int)
    parser.add_argument('--output_c', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'test'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anomaly_ratio', type=float)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--beta_0', type=float, default=0.0001)
    parser.add_argument('--beta_T', type=float, default=0.05)
    parser.add_argument('--mask_scale', type=int, default=5)
    parser.add_argument('--masking', type=str, default='rm')
    parser.add_argument('--masking_k', type=int, default=10)
    parser.add_argument('--only_generate_missing', type=int, default=0)


    config = parser.parse_args()
    args = vars(config)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
