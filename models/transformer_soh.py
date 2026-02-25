import torch
import torch.nn as nn

class BatterySOHTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # SOH is a scalar (0~1)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.embedding(x)  # [batch, seq, d_model]
        x = self.transformer(x) 
        # Global average pooling over sequence
        x = x.mean(dim=1)  
        return self.regressor(x).squeeze(-1)