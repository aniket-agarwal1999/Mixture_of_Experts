import os
from argparse import ArgumentParser
from testing import validation
from train import train
from test import test
from validation import validate


def get_args():
    parser = ArgumentParser(description='Mixture of Experts')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=float, default='mnist')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--validation', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    
    if args.training:
        train(args)
    if args.testing:
        test(args)
    if args.validation:
        validate(args)


if __name__ == '__main__':
    main()