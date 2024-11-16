import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Example model with embedding size of 384


def get_texts_embedding(text_list):
    return embedding_model.encode(text_list)

class SteamReviewsDataset:
    def __init__(self, batch_size = 32, data_path = 'dataset/steam_reviews'):
        self.data_path = data_path
        self.train_loader, self.dev_loader, self.test_loader = load_data(data_path)
        self.data_set = load_all_data(data_path)
        self.batch_size = batch_size


    def load_all_data(self):
        return self.data_set


    def load_data(self):
        train_data = self.train_loader
        dev_data = self.dev_loader
        test_data = self.test_loader

        train_text_list = train_data['text'].tolist()
        dev_text_list = dev_data['text'].tolist()
        test_text_list = test_data['text'].tolist()

        train_text_embeddings = get_texts_embedding(train_text_list)
        dev_text_embeddings = get_texts_embedding(dev_text_list)
        test_text_embeddings = get_texts_embedding(test_text_list)

        train_text_embeddings = torch.tensor(train_text_embeddings)
        dev_text_embeddings = torch.tensor(dev_text_embeddings)
        test_text_embeddings = torch.tensor(test_text_embeddings)

        train_y_true = torch.tensor(train_data['label'].values).float().unsqueeze(1)
        dev_y_true = torch.tensor(dev_data['label'].values).float().unsqueeze(1)
        test_y_true = torch.tensor(test_data['label'].values).float().unsqueeze(1)

        train_dataloader = TensorDataset(train_text_embeddings, train_y_true)
        train_dataloader = DataLoader(train_dataloader, batch_size=self.batch_size, shuffle=True)
        dev_dataloader = TensorDataset(dev_text_embeddings, dev_y_true)
        dev_dataloader = DataLoader(dev_dataloader, batch_size=self.batch_size, shuffle=False)
        test_dataloader = TensorDataset(test_text_embeddings, test_y_true)
        test_dataloader = DataLoader(test_dataloader, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, dev_dataloader, test_dataloader

def load_data(data_path = 'dataset/steam_reviews'):
    testset_path = data_path + '/train_dev_test_split/test.csv'
    devset_path = data_path + '/train_dev_test_split/dev.csv'
    trainset_path = data_path + '/train_dev_test_split/train.csv'

    test_loader = pd.read_csv(testset_path)
    dev_loader = pd.read_csv(devset_path)
    train_loader = pd.read_csv(trainset_path)

    return train_loader, dev_loader, test_loader

def load_all_data(data_path = 'dataset/steam_reviews'):
    data_path = data_path + '/steam_reviews_constructiveness_1.5k.csv'

    data_set = pd.read_csv(data_path)

    return data_set


if __name__ == "__main__":
    steam_dataset = SteamReviewsDataset()
    train_loader, dev_loader, test_loader = steam_dataset.load_data()
    print(train_loader.dataset.tensors[0].shape)