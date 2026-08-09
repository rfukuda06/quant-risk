"""Risk metrics: volatility, Sharpe, Sortino, drawdown, rolling metrics."""

import numpy as np
import pandas as pd

from src.constants import TRADING_DAYS, daily_rf
from src.returns import equity_curve

# A constant return series should have std exactly 0, but floating-point
# summation leaves noise ~1e-19 at some lengths (e.g. std of [0.001]*252),
# which would turn the NaN guards below into ~1e17 Sharpe values. Real daily
# return stds sit many orders of magnitude above this cutoff.
ZERO_TOL = 1e-10


def daily_volatility(returns) -> float:
    """Sample standard deviation (ddof=1) of daily returns."""
    return float(np.asarray(returns, dtype=float).std(ddof=1))


def annualized_volatility(returns) -> float:
    """Daily volatility scaled by sqrt(252); the scaling assumes iid returns."""
    return daily_volatility(returns) * float(np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns, rf_annual: float = 0.0) -> float:
    """Annualized Sharpe: mean daily excess / std daily excess * sqrt(252).

    Accepts a pandas Series or numpy array (the Monte Carlo passes raw rows).
    Returns NaN when volatility is zero — never inf.
    """
    excess = np.asarray(returns, dtype=float) - daily_rf(rf_annual)
    std = excess.std(ddof=1)
    if std < ZERO_TOL:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def downside_deviation(returns, rf_annual: float = 0.0) -> float:
    """Full-sample downside deviation: sqrt(mean(min(r - rf_daily, 0)^2)).

    Averaged over ALL observations. The variant that divides by only the
    negative days inflates Sortino and is rejected by the design spec.
    """
    shortfall = np.minimum(np.asarray(returns, dtype=float) - daily_rf(rf_annual), 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def sortino_ratio(returns, rf_annual: float = 0.0) -> float:
    """Annualized Sortino: mean daily excess / downside deviation * sqrt(252).

    Returns NaN when no observation falls below the daily target — never inf.
    """
    dd = downside_deviation(returns, rf_annual)
    if dd < ZERO_TOL:
        return float("nan")
    excess = np.asarray(returns, dtype=float) - daily_rf(rf_annual)
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))
