import pandas as pd

class YoutubeSpamDataset:
    def __init__(self, data_path = 'dataset/youtube_spam'):
        self.data_path = data_path
        self.train_loader, self.dev_loader, self.test_loader = load_data(data_path)
        self.data_set = load_all_data(data_path)

    def load_data(self):
        return self.train_loader, self.dev_loader, self.test_loader

    def load_all_data(self):
        return self.data_set


def load_data(data_path = 'dataset/youtube_spam'):
    testset_path = data_path + '/train_dev_test_split/test.csv'
    devset_path = data_path + '/train_dev_test_split/dev.csv'
    trainset_path = data_path + '/train_dev_test_split/train.csv'

    test_loader = pd.read_csv(testset_path)
    dev_loader = pd.read_csv(devset_path)
    train_loader = pd.read_csv(trainset_path)

    return train_loader, dev_loader, test_loader

def load_all_data(data_path = 'dataset/youtube_spam'):
    data_path = data_path + '/Youtube-Spam-Dataset.csv'

    data_set = pd.read_csv(data_path)

    return data_set

if __name__ == "__main__":
    utb_dataset = YoutubeSpamDataset()
    train_loader, dev_loader, test_loader = utb_dataset.load_data()
    data_set = utb_dataset.load_all_data()

    print(train_loader.shape, dev_loader.shape, test_loader.shape)
    print(data_set.shape)