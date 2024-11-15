import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))






def main():
    model = LogisticRegression(384)
    example_X = torch.randn(2, 384)

    print("Input shape:", example_X.shape)

    output = model(example_X)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()