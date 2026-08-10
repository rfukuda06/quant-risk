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


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown through time: V_t / P_t - 1 with P_t = max(V_0, ..., V_t).

    The running peak includes the starting value V_0 = 1 (clip), so a
    first-day loss already counts as a drawdown. Always <= 0.
    """
    curve = equity_curve(returns)
    peak = curve.cummax().clip(lower=1.0)
    return curve / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Most negative drawdown reached."""
    return float(drawdown_series(returns).min())


def current_drawdown(returns: pd.Series) -> float:
    """Drawdown at the final observation."""
    return float(drawdown_series(returns).iloc[-1])


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling volatility; the first window-1 points are NaN gaps."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * float(np.sqrt(TRADING_DAYS))


def rolling_sharpe(returns: pd.Series, window: int, rf_annual: float = 0.0) -> pd.Series:
    """Annualized rolling Sharpe; NaN where the rolling std is zero."""
    excess = returns - daily_rf(rf_annual)
    mean = excess.rolling(window, min_periods=window).mean()
    std = excess.rolling(window, min_periods=window).std(ddof=1)
    std = std.mask(std < ZERO_TOL)
    return mean / std * float(np.sqrt(TRADING_DAYS))
