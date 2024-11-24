import pandas as pd
from torch.utils.data import DataLoader
import torch
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Example model with embedding size of 384


def get_texts_embedding(text_list):
    return embedding_model.encode(text_list)


class SteamReviewsDataset:
    def __init__(self, data_set):
        self.data_set = data_set

    def __getitem__(self, index):
        x = self.data_set['text'][index]
        y = self.data_set['label'][index]

        x = get_texts_embedding([x])
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y
    
    def __len__(self):
        return len(self.data_set)


def load_csv_data(data_path):
    testset_path = data_path + '/train_dev_test_split/test.csv'
    devset_path = data_path + '/train_dev_test_split/dev.csv'
    trainset_path = data_path + '/train_dev_test_split/train.csv'

    test_loader = pd.read_csv(testset_path)
    dev_loader = pd.read_csv(devset_path)
    train_loader = pd.read_csv(trainset_path)

    return train_loader, dev_loader, test_loader

def load_all_data(data_path):
    data_path = data_path + '/Youtube-Spam-Dataset.csv'

    data_set = pd.read_csv(data_path)

    return data_set

def load_data(batch_size = 1, data_path = '/userhome/cs2/wang1210/STAT7101_SpamDetection/dataset/steam_reviews'):
    train_loader, dev_loader, test_loader = load_csv_data(data_path)
    test_loader = SteamReviewsDataset(test_loader)
    dev_loader = SteamReviewsDataset(dev_loader)
    train_loader = SteamReviewsDataset(train_loader)

    test_set = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
    dev_set = DataLoader(dev_loader, batch_size=batch_size, shuffle=False)
    train_set = DataLoader(train_loader, batch_size=batch_size, shuffle=False)

    return train_set, dev_set, test_set

def main():
    train_loader, dev_loader, test_loader = load_data()
    for i, (x, y) in enumerate(train_loader):
        print(x.shape, y.shape)
        break

    for i, (x, y) in enumerate(dev_loader):
        print(x.shape, y.shape)
        break

    for i, (x, y) in enumerate(test_loader):
        print(x.shape, y.shape)
        break

    return


if __name__ == "__main__":
    main()