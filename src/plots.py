"""Matplotlib figure builders. Pure: take precomputed data, return Figures.

Figures are built with the Figure constructor, not pyplot, so nothing is
registered in pyplot's global figure manager — Streamlit reruns cannot
accumulate open figures.
"""

import matplotlib

matplotlib.use("Agg")  # off-screen rendering; Streamlit displays figures itself

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def equity_curve_figure(portfolio_curve: pd.Series, benchmark_curve: pd.Series) -> Figure:
    """Growth of $1 in the portfolio vs the benchmark."""
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(portfolio_curve.index, portfolio_curve, label="Portfolio", linewidth=1.5)
    ax.plot(benchmark_curve.index, benchmark_curve, label="Benchmark", linewidth=1.5)
    ax.set_ylabel("Value of $1 invested")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def drawdown_figure(drawdown: pd.Series) -> Figure:
    """Portfolio drawdown through time, shaded below zero."""
    fig = Figure(figsize=(10, 3))
    ax = fig.subplots()
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.4)
    ax.plot(drawdown.index, drawdown, linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def rolling_metric_figure(series: pd.Series, title: str, ylabel: str, as_percent: bool = False) -> Figure:
    """Generic rolling-metric line chart (rolling volatility and rolling Sharpe)."""
    fig = Figure(figsize=(10, 3))
    ax = fig.subplots()
    ax.plot(series.index, series, linewidth=1.2)
    if as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def sharpe_histogram_figure(sharpes: np.ndarray, historical_sharpe: float) -> Figure:
    """Distribution of simulated Sharpe ratios with the historical value marked."""
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.hist(sharpes, bins=60, alpha=0.8)
    ax.axvline(
        historical_sharpe,
        color="crimson",
        linewidth=2,
        label=f"Historical Sharpe = {historical_sharpe:.2f}",
    )
    ax.set_xlabel("Simulated Sharpe ratio")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
