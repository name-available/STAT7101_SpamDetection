import torch
import torch.nn as nn

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
    example_X = torch.randn(2, 384)

    output = model(example_X)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
