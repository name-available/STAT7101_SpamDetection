from torch import nn
import torch
from model.HAN import HAN
from dataloader.dataloader_youtube_spam import load_data as load_data_youtube_spam
from train_test_setting import train_model, test_model
import argparse


def main(args):
    train_loader, dev_loader, test_loader = load_data_youtube_spam(batch_size=1)
    model = HAN(inp_emb_dim=args.text_embedding_dim, hidden_dim=args.hidden_dim, num_classes=2)
    criterion = nn.BCELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_model(model = model, criterion = criterion,
                device=device,
                train_dataloader = train_loader,
                dev_dataloader = dev_loader,
                args = args)

    test_model(model = model,
               device = device,
               criterion = criterion,
               test_dataloader = test_loader,
               args = args)


def get_args():
    args = argparse.ArgumentParser('Training and Evaluation script', add_help=False)
    # data
    args.add_argument('--data_path', default='dataset/youtube_spam', type=str)
    args.add_argument('--batch_size', default=1, type=int)

    # model
    args.add_argument('--best_model_path', default='checkpoints/han.pth', type=str)
    args.add_argument('--text_embedding_dim', default=384, type=int)
    args.add_argument('--hidden_dim', default=128, type=int)

    args.add_argument('--atte_head_num', default=8, type=int)
    args.add_argument('--layer_num', default=6, type=int)
    args.add_argument('--dropout', default=0.5, type=float)

    # training
    args.add_argument('--epochs', default=20, type=int)
    args.add_argument('--eval_per_epochs', default=10, type=int)

    # logging
    args.add_argument('--log_file_name', default="han-utb.log", type=str)

    return args.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)