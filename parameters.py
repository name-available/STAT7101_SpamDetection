import argparse


def get_parameters():
    args = argparse.ArgumentParser('Training and Evaluation script', add_help=False)
    args.add_argument('--epochs', default=500, type=int)
    args.add_argument('--eval_per_epochs', default=50, type=int)
    args.add_argument('--log_file_name', default="LogisticRegression-steam.log", type=str)
    args.add_argument('--best_model_path', default='checkpoints/best_lr_model.pth', type=str)

    return args.parse_args()

def main():
    args = get_parameters()
    print(args)

if __name__ == "__main__":
    main()