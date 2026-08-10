"""Benchmark exposure: correlation and the closed-form CAPM regression.

Covariance is written out by hand so nothing is hidden; scipy/numpy appear
only in the test suite as independent referees.
"""

import numpy as np
import pandas as pd

from src.constants import TRADING_DAYS, daily_rf


def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Sample covariance: sum((x - mean(x)) * (y - mean(y))) / (n - 1)."""
    return float(((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1))


def correlation(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Pearson correlation: Cov(p, b) / (std(p) * std(b)), ddof=1 throughout."""
    p = np.asarray(portfolio, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    return _covariance(p, b) / (p.std(ddof=1) * b.std(ddof=1))


def beta(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0) -> float:
    """CAPM slope on excess returns: Cov(x, y) / Var(x)."""
    rf_d = daily_rf(rf_annual)
    y = np.asarray(portfolio, dtype=float) - rf_d
    x = np.asarray(benchmark, dtype=float) - rf_d
    return _covariance(x, y) / _covariance(x, x)


def alpha_daily(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0) -> float:
    """CAPM intercept in daily units: mean(y) - beta * mean(x)."""
    rf_d = daily_rf(rf_annual)
    y = np.asarray(portfolio, dtype=float) - rf_d
    x = np.asarray(benchmark, dtype=float) - rf_d
    return float(y.mean() - beta(portfolio, benchmark, rf_annual) * x.mean())


def alpha_annualized(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0) -> float:
    """Daily alpha scaled by 252 for display."""
    return alpha_daily(portfolio, benchmark, rf_annual) * TRADING_DAYS


def r_squared(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Share of portfolio variance explained by the benchmark: rho^2.

    Simple-regression identity R^2 = correlation^2; constant rf shifts don't
    affect it, so it is computed on raw returns.
    """
    return correlation(portfolio, benchmark) ** 2
