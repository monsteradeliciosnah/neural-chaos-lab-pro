import numpy as np


def _ensure_xy(xy):
    a = np.asarray(xy, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def lorenz_step(state=None, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    s = _ensure_xy(state if state is not None else [1.0, 1.0, 1.0])
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.stack([x + dx * dt, y + dy * dt, z + dz * dt], axis=1)


def rossler_step(state=None, dt=0.05, a=0.2, b=0.2, c=5.7):
    s = _ensure_xy(state if state is not None else [0.0, 1.0, 0.0])
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.stack([x + dx * dt, y + dy * dt, z + dz * dt], axis=1)


__all__ = ["lorenz_step", "rossler_step"]
