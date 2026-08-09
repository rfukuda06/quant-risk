import pandas as pd
import pytest

from src.constants import TRADING_DAYS
from src.returns import (
    annualized_return,
    best_day,
    cumulative_return,
    equity_curve,
    mean_daily_return,
    worst_day,
)


def series(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_cumulative_return_compounds():
    # 1.10 * 0.95 - 1 = 0.045 exactly
    assert cumulative_return(series([0.10, -0.05])) == pytest.approx(0.045)


def test_cumulative_return_is_not_the_sum():
    # +10% then -10% loses money: 1.10 * 0.90 - 1 = -0.01
    assert cumulative_return(series([0.10, -0.10])) == pytest.approx(-0.01)


def test_equity_curve_tracks_compounded_value():
    curve = equity_curve(series([0.10, -0.05]))
    assert curve.iloc[0] == pytest.approx(1.10)
    assert curve.iloc[-1] == pytest.approx(1.045)


def test_equity_curve_final_value_matches_cumulative_return():
    s = series([0.01, -0.02, 0.03, 0.004])
    assert equity_curve(s).iloc[-1] == pytest.approx(1 + cumulative_return(s))


def test_annualized_return_constant_daily_return():
    # constant r for any N annualizes to exactly (1+r)^252 - 1
    r = 0.001
    assert annualized_return(series([r] * 10)) == pytest.approx((1 + r) ** TRADING_DAYS - 1)


def test_mean_best_worst_day():
    s = series([0.02, -0.03, 0.01])
    assert mean_daily_return(s) == pytest.approx(0.0)
    b_date, b_val = best_day(s)
    w_date, w_val = worst_day(s)
    assert b_val == pytest.approx(0.02) and b_date == s.index[0]
    assert w_val == pytest.approx(-0.03) and w_date == s.index[1]
