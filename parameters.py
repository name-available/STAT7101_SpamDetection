import argparse


def get_parameters():
    args = argparse.ArgumentParser('Training and Evaluation script', add_help=False)
    args.add_argument('--epochs', default=500, type=int)
    args.add_argument('--eval_per_epochs', default=100, type=int)
    args.add_argument('--epochs', default=500, type=int)
    args.add_argument('--epochs', default=500, type=int)
    args.add_argument('--epochs', default=500, type=int)
    args.add_argument('--epochs', default=500, type=int)

    return args
