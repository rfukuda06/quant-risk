"""Shared conventions: annualization factor and risk-free rate handling."""

TRADING_DAYS = 252


def daily_rf(rf_annual: float) -> float:
    """Convert an annual risk-free rate to a daily rate by simple division.

    The geometric conversion (1 + rf) ** (1/252) - 1 differs negligibly at
    realistic rates; the simple convention is pinned in the design spec.
    """
    return rf_annual / TRADING_DAYS


# A constant return series should have std exactly 0, but floating-point
# summation leaves noise (std ~1e-19, variance ~1e-38 at some lengths), which
# would turn NaN guards on zero-denominator ratios into astronomically large
# results instead. Real daily return stds sit many orders of magnitude above
# this cutoff. Compare stds against ZERO_TOL and variances against ZERO_TOL**2.
ZERO_TOL = 1e-10
