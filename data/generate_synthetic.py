"""Generate data/sample_synthetic.csv — correlated portfolio/benchmark returns
with known ground truth. Run from the repo root: python data/generate_synthetic.py

Ground truth (annual): benchmark drift 8%, benchmark vol 16%,
portfolio alpha 2%, beta 1.3, idiosyncratic noise sized for R^2 = 0.75.
Standalone on purpose (no src imports): it documents its own conventions.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SEED = 7
N_DAYS = 756
BENCH_MU_ANNUAL = 0.08
BENCH_SIGMA_ANNUAL = 0.16
TRUE_ALPHA_ANNUAL = 0.02
TRUE_BETA = 1.3
TARGET_R2 = 0.75


def generate(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu_d = BENCH_MU_ANNUAL / TRADING_DAYS
    sigma_d = BENCH_SIGMA_ANNUAL / np.sqrt(TRADING_DAYS)
    alpha_d = TRUE_ALPHA_ANNUAL / TRADING_DAYS
    # solves R2 = (beta * sigma_m)^2 / ((beta * sigma_m)^2 + sigma_eps^2)
    sigma_eps = TRUE_BETA * sigma_d * np.sqrt((1 - TARGET_R2) / TARGET_R2)

    bench = rng.normal(mu_d, sigma_d, n_days)
    port = alpha_d + TRUE_BETA * bench + rng.normal(0, sigma_eps, n_days)
    dates = pd.bdate_range(end="2025-12-31", periods=n_days)
    return pd.DataFrame(
        {
            "date": dates.date,
            "portfolio_return": port.round(6),
            "benchmark_return": bench.round(6),
        }
    )


if __name__ == "__main__":
    generate().to_csv("data/sample_synthetic.csv", index=False)
    print(f"Wrote data/sample_synthetic.csv ({N_DAYS} rows, seed {SEED})")
