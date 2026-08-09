import numpy as np
import pandas as pd
import pytest

from src.constants import TRADING_DAYS
from src.risk import (
    annualized_volatility,
    daily_volatility,
    downside_deviation,
    sharpe_ratio,
    sortino_ratio,
)


def series(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_daily_volatility_is_sample_std():
    s = series([0.01, -0.02, 0.03, 0.00])
    assert daily_volatility(s) == pytest.approx(s.std(ddof=1))


def test_annualized_volatility_scales_by_sqrt_252():
    s = series([0.01, -0.02, 0.03, 0.00])
    assert annualized_volatility(s) == pytest.approx(s.std(ddof=1) * np.sqrt(TRADING_DAYS))


def test_sharpe_ratio_hand_computed():
    s = series([0.01, 0.02, 0.03])
    expected = s.mean() / s.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert sharpe_ratio(s) == pytest.approx(expected)


def test_sharpe_subtracts_risk_free_rate():
    s = series([0.01, 0.02, 0.03])
    excess = s - 0.05 / TRADING_DAYS
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert sharpe_ratio(s, rf_annual=0.05) == pytest.approx(expected)


def test_sharpe_zero_volatility_is_nan_not_inf():
    assert np.isnan(sharpe_ratio(series([0.01, 0.01, 0.01])))
    # at other lengths a constant series has std ~1e-19, not exactly 0.0;
    # the guard must catch floating-point-noise "zero" volatility too
    assert np.isnan(sharpe_ratio(np.full(10, 0.001)))


def test_downside_deviation_hand_computed_over_all_days():
    # shortfalls vs 0: [0, -0.02, 0, -0.01] -> sqrt((0.0004 + 0.0001) / 4)
    s = series([0.01, -0.02, 0.03, -0.01])
    assert downside_deviation(s) == pytest.approx(np.sqrt(0.0005 / 4))


def test_downside_deviation_rejects_negative_days_only_variant():
    # dividing by only the 2 negative days would give sqrt(0.0005 / 2), larger
    s = series([0.01, -0.02, 0.03, -0.01])
    assert downside_deviation(s) < np.sqrt(0.0005 / 2)


def test_sortino_hand_computed():
    s = series([0.01, -0.02, 0.03, -0.01])
    expected = s.mean() / np.sqrt(0.0005 / 4) * np.sqrt(TRADING_DAYS)
    assert sortino_ratio(s) == pytest.approx(expected)


def test_sortino_no_down_days_is_nan_not_inf():
    assert np.isnan(sortino_ratio(series([0.01, 0.02, 0.03])))
