from pathlib import Path

import pandas as pd
import torch
import typer

from .data import lorenz, rossler
from .model import forecast, train_lstm

app = typer.Typer(help="Neural Chaos Lab Pro")


@app.command()
def generate(system: str = "lorenz", n: int = 5000, out: str = "data/series.csv"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = lorenz(n=n) if system == "lorenz" else rossler(n=n)
    df.to_csv(out, index=False)
    print(out)


@app.command()
def train(
    data: str = "data/series.csv",
    epochs: int = 5,
    horizon: int = 10,
    seq_len: int = 50,
    device: str = "cpu",
):
    df = pd.read_csv(data)
    model = train_lstm(
        df, seq_len=seq_len, horizon=horizon, epochs=epochs, device=device
    )
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/lstm.pt")
    print("models/lstm.pt")


@app.command()
def forecast_cmd(
    data: str = "data/series.csv",
    steps: int = 200,
    seq_len: int = 50,
    device: str = "cpu",
):
    import torch

    from .model import LSTMForecaster

    df = pd.read_csv(data)
    model = LSTMForecaster()
    model.load_state_dict(torch.load("models/lstm.pt", map_location=device))
    arr = forecast(model, df, steps=steps, seq_len=seq_len, device=device)
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/forecast.csv")
    pd.DataFrame(arr, columns=["x", "y", "z"]).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    app()
