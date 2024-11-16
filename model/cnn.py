from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=384, embedding_dim=128)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 128), stride=1, padding=(2, 0))
        self.conv2 = nn.Conv2d(1, 128, kernel_size=(4, 128), stride=1, padding=(3, 0))
        self.conv3 = nn.Conv2d(1, 128, kernel_size=(5, 128), stride=1, padding=(4, 0))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = torch.relu(self.conv1(x)).squeeze(3)
        x1 = torch.max_pool1d(x1, kernel_size=x1.size(2)).squeeze(2)
        x2 = torch.relu(self.conv2(x)).squeeze(3)
        x2 = torch.max_pool1d(x2, kernel_size=x2.size(2)).squeeze(2)
        x3 = torch.relu(self.conv3(x)).squeeze(3)
        x3 = torch.max_pool1d(x3, kernel_size=x3.size(2)).squeeze(2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def main():
    model = CNN()

    x = torch.randint(0, 384, (10, 128))

    y = model(x)
    print(y.shape)

if __name__ == "__main__":
    main()

