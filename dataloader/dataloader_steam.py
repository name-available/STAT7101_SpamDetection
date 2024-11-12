import pandas as pd

def load_data(data_path = 'dataset/steam_reviews'):
    testset_path = data_path + '/train_dev_test_split/test.csv'
    devset_path = data_path + '/train_dev_test_split/dev.csv'
    trainset_path = data_path + '/train_dev_test_split/train.csv'

    test_loader = pd.read_csv(testset_path)
    dev_loader = pd.read_csv(devset_path)
    train_loader = pd.read_csv(trainset_path)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = dataloader('dataset/steam_reviews')
    print(train_loader.head())
    print(dev_loader.head())
    print(test_loader.head())
    print(train_loader.shape, dev_loader.shape, test_loader.shape)