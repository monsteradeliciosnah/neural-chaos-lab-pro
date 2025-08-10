import numpy as np
import pandas as pd


def lorenz(n=5000, dt=0.01, sigma=10.0, beta=8 / 3, rho=28.0, x0=(1.0, 1.0, 1.0)):
    x, y, z = x0
    xs = np.empty((n, 3))
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs[i] = [x, y, z]
    return pd.DataFrame(xs, columns=["x", "y", "z"])


def rossler(n=5000, dt=0.01, a=0.2, b=0.2, c=5.7, x0=(1.0, 1.0, 1.0)):
    x, y, z = x0
    xs = np.empty((n, 3))
    for i in range(n):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs[i] = [x, y, z]
    return pd.DataFrame(xs, columns=["x", "y", "z"])
