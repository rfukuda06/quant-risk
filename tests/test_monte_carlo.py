import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (
    estimate_parameters,
    simulate_sharpe_distribution,
    summarize_distribution,
)
from src.risk import sharpe_ratio


@pytest.fixture
def history():
    rng = np.random.default_rng(3)
    return pd.Series(rng.normal(0.0005, 0.01, 250))


def test_estimate_parameters(history):
    mu, sigma = estimate_parameters(history)
    assert mu == pytest.approx(history.mean())
    assert sigma == pytest.approx(history.std(ddof=1))


def test_simulation_is_deterministic_under_fixed_seed(history):
    a = simulate_sharpe_distribution(history, n_sims=200)
    b = simulate_sharpe_distribution(history, n_sims=200)
    assert np.array_equal(a, b)
    # the seed parameter must actually drive the draws
    assert not np.array_equal(a, simulate_sharpe_distribution(history, n_sims=200, seed=99))


def test_simulation_shape_and_finiteness(history):
    sharpes = simulate_sharpe_distribution(history, n_sims=200)
    assert sharpes.shape == (200,)
    assert np.isfinite(sharpes).all()


def test_mean_simulated_sharpe_near_plug_in(history):
    # each simulated history has N=250 days; the annualized Sharpe estimate has
    # std ~ 1.0 at that N, so the mean of 2000 sims should sit well within 0.35
    sharpes = simulate_sharpe_distribution(history, n_sims=2000)
    assert sharpes.mean() == pytest.approx(sharpe_ratio(history), abs=0.35)


def test_summary_statistics(history):
    sharpes = simulate_sharpe_distribution(history, n_sims=500)
    summary = summarize_distribution(sharpes)
    assert summary["p5"] < summary["median"] < summary["p95"]
    assert summary["mean"] == pytest.approx(sharpes.mean())
    assert summary["std"] == pytest.approx(np.std(sharpes, ddof=1))
