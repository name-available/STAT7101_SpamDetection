import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))






def main():
    model = LogisticRegression(384)
    example_X = torch.randn(2, 384)

    print("Input shape:", example_X.shape)

    output = model(example_X)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()