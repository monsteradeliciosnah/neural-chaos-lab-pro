import torch, torch.nn as nn
import numpy as np

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=3, hidden=128, layers=2, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, input_size*horizon)
        self.horizon = horizon
        self.input_size = input_size
    def forward(self, x):
        y,_ = self.lstm(x)
        y = self.fc(y[:,-1,:])
        return y.view(-1, self.horizon, self.input_size)

def make_windows(arr, seq_len=50, horizon=1):
    X=[]; Y=[]
    for i in range(len(arr)-seq_len-horizon):
        X.append(arr[i:i+seq_len])
        Y.append(arr[i+seq_len:i+seq_len+horizon])
    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32)

def train_lstm(series, seq_len=50, horizon=10, epochs=5, lr=1e-3, device="cpu"):
    arr = series.values.astype(np.float32)
    X, Y = make_windows(arr, seq_len, horizon)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    model = LSTMForecaster(input_size=arr.shape[1], hidden=128, layers=2, horizon=horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = lossf(yhat, yb)
            loss.backward(); opt.step()
            total += loss.item()*len(xb)
    return model

@torch.no_grad()
def forecast(model, series, steps=200, seq_len=50, device="cpu"):
    model.eval()
    arr = series.values.astype(np.float32)
    hist = arr[-seq_len:].copy()
    out = []
    cur = torch.from_numpy(hist[None,...]).to(device)
    for _ in range(steps):
        y = model(cur).cpu().numpy()[0, -1]
        out.append(y)
        hist = np.vstack([hist[1:], y])
        cur = torch.from_numpy(hist[None,...]).to(device)
    return np.array(out)
