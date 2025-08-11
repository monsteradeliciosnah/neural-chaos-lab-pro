import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Neural Chaos Lab Pro", layout="wide")
st.title("Neural Chaos Lab Pro — Attractors & Forecasts")

data_path = st.text_input("CSV path", "data/series.csv")
if Path(data_path).exists():
    df = pd.read_csv(data_path)
    st.write(df.head())
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["x"], y=df["y"], z=df["z"], mode="lines", line=dict(width=2)
            )
        ]
    )
    fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Generate a series first.")
