from torch import nn
from model.transformer_lr import TransformerLR
from dataloader.dataloader_youtube_spam import YoutubeSpamDataset
from train_test_setting import train_model, test_model
from parameters import get_parameters


def main(args):
    ytb_dataset = YoutubeSpamDataset()
    train_loader, dev_loader, test_loader = ytb_dataset.load_data()

    model = TransformerLR(args.text_embedding_dim, args.atte_head_num, args.layer_num, args.dropout)
    criterion = nn.BCELoss()
    train_model(model, criterion=criterion, train_dataloader=train_loader, dev_dataloader=dev_loader)
    test_model(model, criterion=criterion, test_dataloader=test_loader)

if __name__ == "__main__":
    args = get_parameters()
    main(args)