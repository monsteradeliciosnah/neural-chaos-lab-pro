from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Series:
    x: np.ndarray
    t: np.ndarray


def _ensure_len3(x0: Sequence[float]) -> Tuple[float, float, float]:
    x = list(x0)
    if len(x) < 3:
        x = (x + [0.0, 0.0, 0.0])[:3]
    return float(x[0]), float(x[1]), float(x[2])


def make_grid(n: int = 100, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    return np.linspace(lo, hi, n, dtype=np.float64)


def lorenz_series(
    x0: Sequence[float] = (1.0, 1.0, 1.0),
    dt: float = 0.01,
    steps: int = 1000,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    rho: float = 28.0,
) -> Series:
    x, y, z = _ensure_len3(x0)
    out = np.zeros((steps, 3), dtype=np.float64)
    t = np.arange(steps, dtype=np.float64) * dt
    for i in range(steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x = x + dt * dx
        y = y + dt * dy
        z = z + dt * dz
        out[i] = (x, y, z)
    return Series(x=out, t=t)


def as_rows(series: Series) -> List[Tuple[float, float, float, float]]:
    rows: List[Tuple[float, float, float, float]] = []
    for i in range(series.x.shape[0]):
        rows.append(
            (
                float(series.t[i]),
                float(series.x[i, 0]),
                float(series.x[i, 1]),
                float(series.x[i, 2]),
            )
        )
    return rows
