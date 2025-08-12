from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import LSTMForecaster, forecast

app = FastAPI(title="Neural Chaos Lab Pro")


class ForecastReq(BaseModel):
    data: str = "data/series.csv"
    steps: int = 200
    seq_len: int = 50


@app.post("/forecast")
def do_forecast(req: ForecastReq):
    if not Path("models/lstm.pt").exists():
        raise HTTPException(400, "Train a model first.")
    df = pd.read_csv(req.data)
    model = LSTMForecaster()
    model.load_state_dict(torch.load("models/lstm.pt", map_location="cpu"))
    out = forecast(model, df, steps=req.steps, seq_len=req.seq_len)
    return {"forecast": out.tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}
