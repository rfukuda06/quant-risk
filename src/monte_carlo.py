"""Monte Carlo sampling-uncertainty analysis of the Sharpe ratio.

Simulates alternate return histories of the SAME length as the historical
sample from Normal(mu_hat, sigma_hat^2) — no other lengths (the sample-size
experiment is an explicit non-goal in the design spec).
"""

import numpy as np

from src.risk import sharpe_ratio

N_SIMS = 10_000
SEED = 42


def estimate_parameters(returns) -> tuple[float, float]:
    """Sample mean and sample std (ddof=1) of historical daily returns."""
    r = np.asarray(returns, dtype=float)
    return float(r.mean()), float(r.std(ddof=1))


def simulate_sharpe_distribution(
    returns, rf_annual: float = 0.0, n_sims: int = N_SIMS, seed: int = SEED
) -> np.ndarray:
    """Sharpe ratios of n_sims simulated histories of length len(returns).

    Each row's Sharpe is computed by the same risk.sharpe_ratio function used
    for the headline metric, so the comparison is apples-to-apples.
    """
    mu, sigma = estimate_parameters(returns)
    rng = np.random.default_rng(seed)
    draws = rng.normal(mu, sigma, size=(n_sims, len(returns)))
    return np.array([sharpe_ratio(row, rf_annual) for row in draws])


def summarize_distribution(sharpes: np.ndarray) -> dict[str, float]:
    """Mean, median, std (ddof=1), and 5th/95th percentiles."""
    return {
        "mean": float(sharpes.mean()),
        "median": float(np.median(sharpes)),
        "std": float(sharpes.std(ddof=1)),
        "p5": float(np.percentile(sharpes, 5)),
        "p95": float(np.percentile(sharpes, 95)),
    }
