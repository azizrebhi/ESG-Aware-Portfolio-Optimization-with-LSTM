import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LSTMReturnPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def prepare_data(returns, window=60):
    scaler = MinMaxScaler()
    scaled_returns = scaler.fit_transform(returns)
    X, y = [], []
    for i in range(window, len(scaled_returns)):
        X.append(scaled_returns[i-window:i])
        y.append(scaled_returns[i])
    X, y = np.array(X), np.array(y)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        scaler
    )

def train_lstm(model, X_tensor, y_tensor, epochs=30, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return model
