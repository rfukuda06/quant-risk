import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plots import (
    drawdown_figure,
    equity_curve_figure,
    rolling_metric_figure,
    sharpe_histogram_figure,
)


def test_all_figure_builders_return_figures():
    dates = pd.bdate_range("2025-01-01", periods=90)
    rng = np.random.default_rng(0)
    curve = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, 90)), index=dates)
    dd = curve / curve.cummax() - 1
    roll = pd.Series(rng.normal(0.15, 0.02, 90), index=dates)

    assert isinstance(equity_curve_figure(curve, curve * 0.9), Figure)
    assert isinstance(drawdown_figure(dd), Figure)
    assert isinstance(rolling_metric_figure(roll, "title", "ylabel", as_percent=True), Figure)
    assert isinstance(sharpe_histogram_figure(rng.normal(1.0, 0.5, 1000), 1.1), Figure)
