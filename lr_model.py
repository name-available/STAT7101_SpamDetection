from torch import nn
from model.logistic_regression_model import LogisticRegressionModel
from dataloader.dataloader_youtube_spam import YoutubeSpamDataset
from train_test_setting import train_model, test_model

def main():
    ytb_dataset = YoutubeSpamDataset()
    train_loader, dev_loader, test_loader = ytb_dataset.load_data()
    print(train_loader.dataset.tensors[0].shape)
    print(dev_loader.dataset.tensors[0].shape)
    print(test_loader.dataset.tensors[0].shape)

    
    model = LogisticRegressionModel(384)
    criterion = nn.BCELoss()
    train_model(model, criterion=criterion, train_dataloader=train_loader, dev_dataloader=dev_loader)
    test_model(model, criterion=criterion, test_dataloader=test_loader)

if __name__ == "__main__":
    main()