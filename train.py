import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.transformer_soh import BatterySOHTransformer

import os

os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

from data.synthetic_battery_data import generate_china_battery_data
df = generate_china_battery_data(n_batteries=20, cycles_per_battery=100)
df.to_csv("data/synthetic_battery.csv", index=False)

def extract_features(df):
    features = df[['temp', 'dod', 'is_fast_charge', 'cycle']].values
    targets = df['soh'].values
    return features, targets

X, y = extract_features(df)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(X, y, seq_len=50):
    X_seq, y_seq = [], []
    for bat_id in df['battery_id'].unique():
        bat_mask = df['battery_di'] == bat_id
        X_bat = X_scaled[bat_mask]
        y_bat = y[bat_mask]
        if len(X_bat) >= seq_len:
            for i in range(len(X_bat) - seq_len + 1):
                X_seq.append(X_bat[i:i+seq_len])
                y_seq.append(y_bat[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, seq_len=30)

split = int(0.8 * len(X_seq))
X_train, X_testr = X_seq[:split], X_seq[split:]
y_train, y_test =  y_seq[:split], y_seq[split:]

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    ),
    batch_size=16, shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BatterySOHTransformer(input_dim=4).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
for epoch in range(50):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "checkpoints/soh_model.pth")
print("Model saved to checkpoints/soh_model.pth")

dummy_input = torch.randn(1, 30, 4).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "checkpoints/soh_model.onnx",
    input_names=['input'],
    output_names=["soh"],
    dynamic_axes={"input": {0: "batch"}, "soh": {0: "batch"}}
)

print("ONNX model exported to checkpoints/soh_model.onnx")