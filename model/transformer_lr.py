import torch
import torch.nn as nn
from torch import optim

class TransformerLR(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=6, dropout=0.5):
        super(TransformerLR, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return torch.sigmoid(self.linear(x))
    
def main():
    model = TransformerLR(384)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example data
    example_X = torch.randn(10, 384)  # Batch size of 10
    example_y = torch.randint(0, 2, (10, 1)).float()  # Binary labels

    # Training loop
    model.train()
    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()
        outputs = model(example_X)
        loss = criterion(outputs, example_y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
