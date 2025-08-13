from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import LSTMForecaster, forecast

app = FastAPI(title="Neural Chaos Lab Pro")


class ForecastRequest(BaseModel):
    series: list[float]
    horizon: int = 10


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast")
def forecast_route(req: ForecastRequest) -> dict:
    try:
        model = LSTMForecaster()
        yhat = forecast(model, req.series, horizon=req.horizon)
        return {"forecast": yhat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
