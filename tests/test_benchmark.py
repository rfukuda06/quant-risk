import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.benchmark import alpha_annualized, alpha_daily, beta, correlation, r_squared
from src.constants import TRADING_DAYS


@pytest.fixture
def market_like():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.0004, 0.01, 500))
    port = 0.0001 + 1.4 * bench + pd.Series(rng.normal(0, 0.005, 500))
    return port, bench


def test_beta_matches_scipy_linregress(market_like):
    port, bench = market_like
    assert beta(port, bench) == pytest.approx(stats.linregress(bench, port).slope, abs=1e-12)


def test_alpha_matches_scipy_linregress(market_like):
    port, bench = market_like
    ref = stats.linregress(bench, port)
    assert alpha_daily(port, bench) == pytest.approx(ref.intercept, abs=1e-12)
    assert alpha_annualized(port, bench) == pytest.approx(ref.intercept * TRADING_DAYS, abs=1e-9)


def test_correlation_matches_numpy_corrcoef(market_like):
    port, bench = market_like
    assert correlation(port, bench) == pytest.approx(np.corrcoef(port, bench)[0, 1], abs=1e-12)


def test_r_squared_equals_rvalue_squared(market_like):
    port, bench = market_like
    assert r_squared(port, bench) == pytest.approx(stats.linregress(bench, port).rvalue ** 2, abs=1e-12)


def test_constant_rf_shift_moves_only_alpha(market_like):
    # subtracting a constant rf_daily from both series cannot change covariances
    port, bench = market_like
    assert beta(port, bench, rf_annual=0.05) == pytest.approx(beta(port, bench), abs=1e-12)
    assert alpha_daily(port, bench, rf_annual=0.05) != pytest.approx(alpha_daily(port, bench))


def test_perfectly_scaled_portfolio():
    bench = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    port = 2 * bench
    assert beta(port, bench) == pytest.approx(2.0)
    assert r_squared(port, bench) == pytest.approx(1.0)


def test_constant_benchmark_degenerate_cases_are_nan_not_error():
    port = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 60))
    for const in (0.0, 0.001):  # exact-zero variance and FP-noise variance
        flat = pd.Series([const] * 60)
        assert np.isnan(beta(port, flat))
        assert np.isnan(correlation(port, flat))
        assert np.isnan(r_squared(port, flat))
