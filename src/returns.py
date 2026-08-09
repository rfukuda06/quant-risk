"""Performance metrics: compounding, annualization, equity curve."""

import pandas as pd

from src.constants import TRADING_DAYS


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return: (1 + r_1)(1 + r_2)...(1 + r_n) - 1."""
    return float((1 + returns).prod() - 1)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Growth of $1: V_t = prod_{s<=t}(1 + r_s), starting from V_0 = 1."""
    return (1 + returns).cumprod()


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return (CAGR): (1 + total)^(252/N) - 1."""
    return float((1 + cumulative_return(returns)) ** (TRADING_DAYS / len(returns)) - 1)


def mean_daily_return(returns: pd.Series) -> float:
    """Arithmetic mean of daily returns."""
    return float(returns.mean())


def best_day(returns: pd.Series) -> tuple[pd.Timestamp, float]:
    """Date and value of the largest daily return."""
    return returns.idxmax(), float(returns.max())


def worst_day(returns: pd.Series) -> tuple[pd.Timestamp, float]:
    """Date and value of the smallest daily return."""
    return returns.idxmin(), float(returns.min())
