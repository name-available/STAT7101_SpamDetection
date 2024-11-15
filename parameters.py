import argparse


def get_parameters():
    args = argparse.ArgumentParser('Training and Evaluation script', add_help=False)
    # data
    args.add_argument('--data_path', default='dataset/youtube_spam', type=str)
    args.add_argument('--batch_size', default=32, type=int)

    # model
    args.add_argument('--best_model_path', default='checkpoints/lr-4.pth', type=str)
    args.add_argument('--text_embedding_dim', default=384, type=int)
    args.add_argument('--atte_head_num', default=8, type=int)
    args.add_argument('--layer_num', default=6, type=int)
    args.add_argument('--dropout', default=0.5, type=float)

    # training
    args.add_argument('--epochs', default=500, type=int)
    args.add_argument('--eval_per_epochs', default=1, type=int)

    # logging
    args.add_argument('--log_file_name', default="lr-4-steam.log", type=str)

    return args.parse_args()

def main():
    args = get_parameters()
    print(args)

if __name__ == "__main__":
    main()