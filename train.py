import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load data
df = pd.read_csv("data1/landmarks/landmarks.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("model", exist_ok=True)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Dataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# MLP Model
class RPS_MLP(nn.Module):
    def __init__(self):
        super(RPS_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.model(x)

model = RPS_MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            predicted = torch.argmax(test_output, dim=1)
            accuracy = (predicted == y_test).float().mean().item()
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f} — Test Accuracy: {accuracy*100:.2f}%")

# Save model
torch.save(model.state_dict(), "model/rps_model.pth")
print("\nModel saved to model/rps_model.pth")
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

# Check class distribution
from collections import Counter
print(Counter(predicted.numpy()))
print(Counter(y_test.numpy()))