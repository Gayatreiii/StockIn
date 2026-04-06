import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    close = df["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(len(close)).mean()

    std = close.rolling(20).std()
    bb_upper = ma20 + 2 * std
    bb_lower = ma20 - 2 * std

    fig = go.Figure()

    # Bollinger band fill
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_upper, line=dict(width=0),
        showlegend=False, name="BB Upper"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_lower, fill="tonexty",
        fillcolor="rgba(99,102,241,0.08)", line=dict(width=0),
        showlegend=False, name="BB Lower"
    ))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",
        decreasing_fillcolor="#ef4444",
        name=ticker.replace(".NS", ""),
        whiskerwidth=0.3,
    ))

    # MA lines
    fig.add_trace(go.Scatter(
        x=df.index, y=ma20,
        line=dict(color="#6366f1", width=1.5),
        name="MA20"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=ma50,
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        name="MA50"
    ))

    fig.update_layout(
        paper_bgcolor="#12121a",
        plot_bgcolor="#12121a",
        font=dict(color="#9ca3af", family="Inter"),
        xaxis=dict(
            gridcolor="#1e1e2e", showgrid=True,
            rangeslider=dict(visible=False),
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="#1e1e2e", showgrid=True,
            tickfont=dict(size=11),
            tickprefix="₹",
        ),
        legend=dict(
            bgcolor="#12121a", bordercolor="#1e1e2e",
            font=dict(size=11),
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0
        ),
        margin=dict(l=0, r=0, t=36, b=0),
        height=320,
    )
    fig.update_xaxes(showspikes=True, spikecolor="#2d2d3d", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor="#2d2d3d", spikethickness=1)
    return fig


def plot_indicators_chart(df: pd.DataFrame) -> go.Figure:
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        line=dict(color="#6366f1", width=2),
        name="RSI"
    ))
    fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=1))
    fig.add_hline(y=30, line=dict(color="#22c55e", dash="dash", width=1))
    fig.add_hline(y=50, line=dict(color="#4b5563", dash="dot", width=1))

    fig.update_layout(
        paper_bgcolor="#12121a", plot_bgcolor="#12121a",
        font=dict(color="#9ca3af", family="Inter"),
        xaxis=dict(gridcolor="#1e1e2e", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#1e1e2e", range=[0, 100]),
        margin=dict(l=0, r=0, t=20, b=0),
        height=160,
        showlegend=False,
    )
    return fig
