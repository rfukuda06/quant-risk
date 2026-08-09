"""Shared conventions: annualization factor and risk-free rate handling."""

TRADING_DAYS = 252


def daily_rf(rf_annual: float) -> float:
    """Convert an annual risk-free rate to a daily rate by simple division.

    The geometric conversion (1 + rf) ** (1/252) - 1 differs negligibly at
    realistic rates; the simple convention is pinned in the design spec.
    """
    return rf_annual / TRADING_DAYS
