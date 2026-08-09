# Quantitative Risk & Performance Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard that analyzes a CSV of daily portfolio/benchmark returns for performance, risk, benchmark exposure, and Monte Carlo Sharpe-ratio uncertainty — every formula hand-rolled and test-verified.

**Architecture:** Pure analysis modules in `src/` (no Streamlit imports), a thin `app.py` UI, committed sample datasets with provenance scripts, and a three-layer pytest suite (hand-computed known answers, scipy/pandas referee cross-checks, behavioral/edge tests). Source of truth: `docs/superpowers/specs/2026-08-09-quant-risk-analyzer-design.md`.

**Tech Stack:** Python 3.13, uv-managed venv. Runtime: streamlit, pandas, numpy, matplotlib. Dev: pytest, scipy. `yfinance` used once by a provenance script, never a dependency.

**Git:** Remote `origin` is already configured (github.com/rfukuda06/quant-risk). The user has authorized pushing at milestones — push steps appear at the end of Tasks 8, 10, and 12. Commit after every task; end every commit message with the trailer shown in Task 1.

**Conventions pinned by the spec (apply everywhere):** returns are decimals; every std is `ddof=1`; annualization uses 252; `rf_daily = rf_annual / 252`; zero-denominator ratios return `NaN`, never `inf`.

---

### Task 1: Project scaffold

**Files:**
- Create: `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `conftest.py`, `src/__init__.py`, `src/constants.py`, `tests/test_constants.py`

- [ ] **Step 1: Create environment and config files**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

`requirements.txt`:
```
streamlit
pandas
numpy
matplotlib
```

`requirements-dev.txt`:
```
pytest
scipy
```

`conftest.py` — empty file at repo root. Its presence makes pytest add the repo root to `sys.path`, so tests can `import src.*` without installing the package.

`src/__init__.py` — empty file.

- [ ] **Step 2: Create the venv and install**

Run:
```bash
uv venv .venv
uv pip install -r requirements.txt -r requirements-dev.txt
```
Expected: both commands succeed; `uv pip` auto-detects `.venv`.

- [ ] **Step 3: Write the failing test**

`tests/test_constants.py`:
```python
from src.constants import TRADING_DAYS, daily_rf


def test_trading_days():
    assert TRADING_DAYS == 252


def test_daily_rf_is_simple_division():
    assert daily_rf(0.0504) == 0.0002
    assert daily_rf(0.0) == 0.0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_constants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.constants'`

- [ ] **Step 5: Write the implementation**

`src/constants.py`:
```python
"""Shared conventions: annualization factor and risk-free rate handling."""

TRADING_DAYS = 252


def daily_rf(rf_annual: float) -> float:
    """Convert an annual risk-free rate to a daily rate by simple division.

    The geometric conversion (1 + rf) ** (1/252) - 1 differs negligibly at
    realistic rates; the simple convention is pinned in the design spec.
    """
    return rf_annual / TRADING_DAYS
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_constants.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt requirements-dev.txt conftest.py src/ tests/
git commit -m "chore: scaffold project with uv env, pytest, and shared constants" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Use this same trailer on every commit in this plan.

---

### Task 2: returns.py — compounding, CAGR, equity curve

**Files:**
- Create: `src/returns.py`
- Test: `tests/test_returns.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_returns.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_returns.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.returns'`

- [ ] **Step 3: Write the implementation**

`src/returns.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_returns.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/returns.py tests/test_returns.py
git commit -m "feat: add performance metrics (cumulative, CAGR, equity curve)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: risk.py part 1 — volatility, Sharpe, Sortino

**Files:**
- Create: `src/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_risk.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_risk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.risk'`

- [ ] **Step 3: Write the implementation**

`src/risk.py`:
```python
"""Risk metrics: volatility, Sharpe, Sortino, drawdown, rolling metrics."""

import numpy as np
import pandas as pd

from src.constants import TRADING_DAYS, daily_rf
from src.returns import equity_curve


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
    if std == 0:
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
    if dd == 0:
        return float("nan")
    excess = np.asarray(returns, dtype=float) - daily_rf(rf_annual)
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_risk.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/risk.py tests/test_risk.py
git commit -m "feat: add volatility, Sharpe, and full-sample Sortino" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: risk.py part 2 — drawdown and rolling metrics

**Files:**
- Modify: `src/risk.py` (append functions)
- Modify: `tests/test_risk.py` (append tests)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_risk.py` (the `series` helper already exists there):
```python
from src.risk import (
    current_drawdown,
    drawdown_series,
    max_drawdown,
    rolling_sharpe,
    rolling_volatility,
)


def test_max_drawdown_crafted_series():
    # curve doubles to 2.0 then falls to 1.0 -> max drawdown exactly -50%
    assert max_drawdown(series([1.0, -0.5])) == pytest.approx(-0.5)


def test_drawdown_never_positive_and_zero_at_new_peaks():
    s = series([0.02, -0.01, 0.03, -0.05, 0.01])
    dd = drawdown_series(s)
    assert (dd <= 1e-12).all()
    assert dd.iloc[0] == pytest.approx(0.0)  # first day sets a new peak
    assert dd.iloc[2] == pytest.approx(0.0)  # recovers past the old peak


def test_drawdown_measured_from_initial_dollar():
    # a first-day loss is already a drawdown relative to the starting $1 (V_0 = 1)
    dd = drawdown_series(series([-0.10, 0.05]))
    assert dd.iloc[0] == pytest.approx(-0.10)


def test_current_drawdown_is_last_value():
    s = series([0.02, -0.01, 0.03, -0.05, 0.01])
    assert current_drawdown(s) == pytest.approx(drawdown_series(s).iloc[-1])


def test_rolling_volatility_window_and_gaps():
    s = series(list(np.linspace(-0.01, 0.01, 40)))
    roll = rolling_volatility(s, window=30)
    assert roll.iloc[:29].isna().all()  # no partial windows
    assert roll.iloc[29] == pytest.approx(s.iloc[:30].std(ddof=1) * np.sqrt(TRADING_DAYS))


def test_rolling_sharpe_matches_full_sample_on_exact_window():
    s = series(list(np.random.default_rng(0).normal(0.001, 0.01, 60)))
    assert rolling_sharpe(s, window=60).iloc[-1] == pytest.approx(sharpe_ratio(s))


def test_rolling_sharpe_zero_std_window_is_nan():
    roll = rolling_sharpe(series([0.01] * 35), window=30)
    assert roll.iloc[29:].isna().all()
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `.venv/bin/pytest tests/test_risk.py -v`
Expected: ImportError — `cannot import name 'current_drawdown' from 'src.risk'`

- [ ] **Step 3: Append the implementation**

Append to `src/risk.py`:
```python
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
    std = excess.rolling(window, min_periods=window).std(ddof=1).replace(0.0, np.nan)
    return mean / std * float(np.sqrt(TRADING_DAYS))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_risk.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add src/risk.py tests/test_risk.py
git commit -m "feat: add drawdown and rolling risk metrics" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: benchmark.py — correlation and closed-form CAPM

**Files:**
- Create: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.benchmark'`

- [ ] **Step 3: Write the implementation**

`src/benchmark.py`:
```python
"""Benchmark exposure: correlation and the closed-form CAPM regression.

Covariance is written out by hand so nothing is hidden; scipy/numpy appear
only in the test suite as independent referees.
"""

import numpy as np
import pandas as pd

from src.constants import TRADING_DAYS, daily_rf


def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Sample covariance: sum((x - mean(x)) * (y - mean(y))) / (n - 1)."""
    return float(((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1))


def correlation(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Pearson correlation: Cov(p, b) / (std(p) * std(b)), ddof=1 throughout."""
    p = np.asarray(portfolio, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    return _covariance(p, b) / (p.std(ddof=1) * b.std(ddof=1))


def beta(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0) -> float:
    """CAPM slope on excess returns: Cov(x, y) / Var(x)."""
    rf_d = daily_rf(rf_annual)
    y = np.asarray(portfolio, dtype=float) - rf_d
    x = np.asarray(benchmark, dtype=float) - rf_d
    return _covariance(x, y) / _covariance(x, x)


def alpha_daily(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0) -> float:
    """CAPM intercept in daily units: mean(y) - beta * mean(x)."""
    rf_d = daily_rf(rf_annual)
    y = np.asarray(portfolio, dtype=float) - rf_d
    x = np.asarray(benchmark, dtype=float) - rf_d
    return float(y.mean() - beta(portfolio, benchmark, rf_annual) * x.mean())


def alpha_annualized(portfolio: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0) -> float:
    """Daily alpha scaled by 252 for display."""
    return alpha_daily(portfolio, benchmark, rf_annual) * TRADING_DAYS


def r_squared(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Share of portfolio variance explained by the benchmark: rho^2.

    Simple-regression identity R^2 = correlation^2; constant rf shifts don't
    affect it, so it is computed on raw returns.
    """
    return correlation(portfolio, benchmark) ** 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_benchmark.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "feat: add correlation and hand-rolled CAPM (beta, alpha, R2)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: monte_carlo.py — Sharpe sampling-uncertainty simulation

**Files:**
- Create: `src/monte_carlo.py`
- Test: `tests/test_monte_carlo.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_monte_carlo.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_monte_carlo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.monte_carlo'`

- [ ] **Step 3: Write the implementation**

`src/monte_carlo.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_monte_carlo.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/monte_carlo.py tests/test_monte_carlo.py
git commit -m "feat: add Monte Carlo Sharpe uncertainty simulation" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: data_loader.py — CSV validation

**Files:**
- Create: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing tests** (one per spec §5 rule)

`tests/test_data_loader.py`:
```python
import io

import pandas as pd
import pytest

from src.data_loader import ValidationError, load_returns


def make_csv(rows, header="date,portfolio_return,benchmark_return"):
    return io.StringIO(header + "\n" + "\n".join(rows))


def valid_rows(n=70):
    dates = pd.bdate_range("2025-01-01", periods=n)
    return [f"{d.date()},0.001,0.002" for d in dates]


def test_valid_file_loads_clean():
    df, messages = load_returns(make_csv(valid_rows()))
    assert list(df.columns) == ["portfolio", "benchmark"]
    assert df.index.is_monotonic_increasing
    assert len(df) == 70
    assert messages == []


def test_missing_column_is_error():
    with pytest.raises(ValidationError, match="benchmark_return"):
        load_returns(make_csv(["2025-01-02,0.001"], header="date,portfolio_return"))


def test_case_insensitive_columns_and_extra_column_note():
    header = "Date,Portfolio_Return,BENCHMARK_RETURN,notes"
    rows = [r + ",x" for r in valid_rows()]
    df, messages = load_returns(make_csv(rows, header=header))
    assert len(df) == 70
    assert any("extra" in m.text.lower() for m in messages)


def test_unparseable_date_is_error():
    rows = valid_rows()
    rows[5] = "not-a-date,0.001,0.002"
    with pytest.raises(ValidationError, match="not-a-date"):
        load_returns(make_csv(rows))


def test_non_numeric_return_is_error():
    rows = valid_rows()
    rows[3] = rows[3].replace(",0.001,", ",abc,")
    with pytest.raises(ValidationError, match="abc"):
        load_returns(make_csv(rows))


def test_duplicate_dates_error():
    rows = valid_rows()
    rows[10] = rows[9]
    with pytest.raises(ValidationError, match="Duplicate"):
        load_returns(make_csv(rows))


def test_unsorted_dates_sorted_with_info():
    rows = valid_rows()
    rows.reverse()
    df, messages = load_returns(make_csv(rows))
    assert df.index.is_monotonic_increasing
    assert any(m.severity == "info" and "sorted" in m.text.lower() for m in messages)


def test_missing_values_dropped_with_warning():
    rows = valid_rows()
    rows[7] = rows[7].replace(",0.002", ",")  # blank benchmark cell
    df, messages = load_returns(make_csv(rows))
    assert len(df) == 69
    assert any(m.severity == "warning" and "Dropped 1" in m.text for m in messages)


def test_percent_scale_warning():
    dates = pd.bdate_range("2025-01-01", periods=70)
    rows = [f"{d.date()},0.8,1.1" for d in dates]  # clearly percent-scale values
    _, messages = load_returns(make_csv(rows))
    assert any("percent" in m.text.lower() for m in messages)


def test_extreme_return_warning():
    rows = valid_rows()
    rows[4] = rows[4].replace(",0.001,", ",0.75,")
    _, messages = load_returns(make_csv(rows))
    assert any("large daily return" in m.text.lower() for m in messages)


def test_too_few_rows_error():
    with pytest.raises(ValidationError, match="at least 60"):
        load_returns(make_csv(valid_rows(50)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_loader'`

- [ ] **Step 3: Write the implementation**

`src/data_loader.py`:
```python
"""CSV loading and validation for portfolio/benchmark return files.

Pure (no Streamlit): returns a clean DataFrame plus info/warning messages;
raises ValidationError with a user-facing message when analysis is impossible.
"""

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = {"date", "portfolio_return", "benchmark_return"}
MIN_ROWS = 60
PERCENT_SNIFF_THRESHOLD = 0.05
EXTREME_RETURN_THRESHOLD = 0.5


class ValidationError(Exception):
    """The uploaded CSV cannot be analyzed; the message is user-facing."""


@dataclass
class LoaderMessage:
    severity: str  # "info" or "warning"
    text: str


def load_returns(file_or_buffer) -> tuple[pd.DataFrame, list[LoaderMessage]]:
    """Parse and validate a returns CSV per the design spec's §5 rules.

    Output: DataFrame with ascending DatetimeIndex and float64 columns
    'portfolio' and 'benchmark', plus a list of LoaderMessages.
    """
    messages: list[LoaderMessage] = []
    raw = pd.read_csv(file_or_buffer)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValidationError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            "Expected columns: date, portfolio_return, benchmark_return."
        )
    extra = set(raw.columns) - REQUIRED_COLUMNS
    if extra:
        messages.append(
            LoaderMessage("info", f"Ignoring extra column(s): {', '.join(sorted(extra))}.")
        )

    dates = pd.to_datetime(raw["date"], errors="coerce")
    bad_dates = raw.loc[dates.isna() & raw["date"].notna(), "date"]
    if not bad_dates.empty:
        examples = ", ".join(str(v) for v in bad_dates.head(3))
        raise ValidationError(f"Unparseable date(s), e.g.: {examples}.")

    parsed = {}
    for col, name in [("portfolio_return", "portfolio"), ("benchmark_return", "benchmark")]:
        coerced = pd.to_numeric(raw[col], errors="coerce")
        garbage = raw.loc[coerced.isna() & raw[col].notna(), col]
        if not garbage.empty:
            examples = ", ".join(str(v) for v in garbage.head(3))
            raise ValidationError(f"Non-numeric value(s) in {col}, e.g.: {examples}.")
        parsed[name] = coerced

    df = pd.DataFrame(parsed)
    df.index = pd.DatetimeIndex(dates)
    incomplete = df.index.isna() | df.isna().any(axis=1)
    if incomplete.any():
        messages.append(
            LoaderMessage("warning", f"Dropped {int(incomplete.sum())} row(s) with missing values.")
        )
        df = df.loc[~incomplete]

    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].strftime("%Y-%m-%d")
        raise ValidationError(f"Duplicate date(s): {', '.join(dupes[:3])}.")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        messages.append(
            LoaderMessage("info", "Rows were not in chronological order; sorted by date.")
        )

    if len(df) < MIN_ROWS:
        raise ValidationError(f"Need at least {MIN_ROWS} rows of data; got {len(df)}.")

    all_values = pd.concat([df["portfolio"], df["benchmark"]])
    if all_values.abs().median() > PERCENT_SNIFF_THRESHOLD:
        messages.append(
            LoaderMessage(
                "warning",
                "Values look like percentages, not decimals (median |return| > 5%). "
                "Expected decimals, e.g. 0.0021 for 0.21%. Not auto-converting.",
            )
        )

    extreme = df[(df.abs() > EXTREME_RETURN_THRESHOLD).any(axis=1)].index.strftime("%Y-%m-%d")
    if len(extreme) > 0:
        messages.append(
            LoaderMessage(
                "warning",
                f"Suspiciously large daily return(s) (>50%) on: {', '.join(extreme[:3])}. "
                "Check the data.",
            )
        )

    return df.astype(float), messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: add CSV loader with spec'd validation rules" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Sample data — synthetic generator, real fetch, end-to-end recovery test

**Files:**
- Create: `data/generate_synthetic.py`, `data/fetch_real.py`, `data/sample_synthetic.csv` (generated), `data/sample_real.csv` (fetched)
- Test: `tests/test_sample_data.py`

- [ ] **Step 1: Write the synthetic generator**

`data/generate_synthetic.py`:
```python
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
```

- [ ] **Step 2: Generate the synthetic CSV**

Run (from repo root): `.venv/bin/python data/generate_synthetic.py`
Expected: `Wrote data/sample_synthetic.csv (756 rows, seed 7)`

- [ ] **Step 3: Write the real-data fetch script**

`data/fetch_real.py`:
```python
"""One-time script that produced data/sample_real.csv. yfinance is NOT a
project dependency; to re-run: uv run --with yfinance python data/fetch_real.py

Portfolio: equal-weight daily mean of AAPL, MSFT, NVDA, AMZN daily returns
(daily-rebalanced equal weighting). Benchmark: SPY. Adjusted closes,
2022-01-01 through 2024-12-31.
"""

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN"]
BENCHMARK = "SPY"
START, END = "2022-01-01", "2025-01-01"  # END is exclusive -> through 2024-12-31

prices = yf.download(TICKERS + [BENCHMARK], start=START, end=END, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()
out = returns[TICKERS].mean(axis=1).to_frame("portfolio_return")
out["benchmark_return"] = returns[BENCHMARK]
out.index.name = "date"
out.round(6).to_csv("data/sample_real.csv")
print(f"Wrote data/sample_real.csv ({len(out)} rows, {out.index[0].date()} to {out.index[-1].date()})")
```

- [ ] **Step 4: Fetch the real data (needs internet, once)**

Run (from repo root): `uv run --with yfinance --no-project python data/fetch_real.py`
Expected: `Wrote data/sample_real.csv (~751 rows, 2022-01-04 to 2024-12-31)` (row count in the low 750s).
If yfinance's return shape has changed (its API drifts), inspect `prices.columns` and adjust the `["Close"]` selection; if Yahoo blocks the fetch entirely, report to the user rather than substituting fake data.

- [ ] **Step 5: Write the failing end-to-end tests**

`tests/test_sample_data.py`:
```python
"""End-to-end: the pipeline recovers the generator's ground truth, and the
committed sample CSVs load cleanly through the validator."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from generate_synthetic import TRUE_ALPHA_ANNUAL, TRUE_BETA, generate  # noqa: E402

from src.benchmark import alpha_annualized, beta, r_squared  # noqa: E402
from src.data_loader import load_returns  # noqa: E402


def test_pipeline_recovers_ground_truth_at_large_n():
    # At N=50,000 the standard errors are ~0.003 (beta) and ~0.9% (annual alpha),
    # so these tolerances are several sigma wide yet still catch real formula bugs.
    df = generate(n_days=50_000, seed=123)
    port = pd.Series(df["portfolio_return"].values)
    bench = pd.Series(df["benchmark_return"].values)
    assert beta(port, bench) == pytest.approx(TRUE_BETA, abs=0.02)
    assert alpha_annualized(port, bench) == pytest.approx(TRUE_ALPHA_ANNUAL, abs=0.03)
    assert r_squared(port, bench) == pytest.approx(0.75, abs=0.02)


def test_committed_synthetic_sample_loads_clean():
    df, messages = load_returns(REPO_ROOT / "data" / "sample_synthetic.csv")
    assert len(df) == 756
    assert messages == []


def test_committed_real_sample_loads_clean():
    df, messages = load_returns(REPO_ROOT / "data" / "sample_real.csv")
    assert len(df) > 700
    assert [m for m in messages if m.severity == "warning"] == []
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/pytest tests/test_sample_data.py -v`
Expected: 3 passed (the "failing first" step here is the CSVs not existing — if you wrote tests before Steps 2/4 they fail on missing files; either order is fine as long as you see them pass now). If the large-N alpha assertion misses by a hair, widen `abs=` to 0.04 — the tolerance is a sampling-noise guard, not a spec number.

- [ ] **Step 7: Commit**

```bash
git add data/ tests/test_sample_data.py
git commit -m "feat: add synthetic and real sample datasets with provenance scripts" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: MILESTONE PUSH — core engine complete**

Run: `.venv/bin/pytest -v` (full suite, expect all green: ~43 tests), then:
```bash
git push
```

---

### Task 9: plots.py — matplotlib figure builders

**Files:**
- Create: `src/plots.py`
- Test: `tests/test_plots.py`

- [ ] **Step 1: Write the failing smoke test**

`tests/test_plots.py`:
```python
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plots import (
    drawdown_figure,
    equity_curve_figure,
    rolling_metric_figure,
    sharpe_histogram_figure,
)


def test_all_figure_builders_return_figures():
    dates = pd.bdate_range("2025-01-01", periods=90)
    rng = np.random.default_rng(0)
    curve = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, 90)), index=dates)
    dd = curve / curve.cummax() - 1
    roll = pd.Series(rng.normal(0.15, 0.02, 90), index=dates)

    assert isinstance(equity_curve_figure(curve, curve * 0.9), Figure)
    assert isinstance(drawdown_figure(dd), Figure)
    assert isinstance(rolling_metric_figure(roll, "title", "ylabel", as_percent=True), Figure)
    assert isinstance(sharpe_histogram_figure(rng.normal(1.0, 0.5, 1000), 1.1), Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.plots'`

- [ ] **Step 3: Write the implementation**

`src/plots.py`:
```python
"""Matplotlib figure builders. Pure: take precomputed data, return Figures."""

import matplotlib

matplotlib.use("Agg")  # off-screen rendering; Streamlit displays figures itself

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def equity_curve_figure(portfolio_curve: pd.Series, benchmark_curve: pd.Series) -> Figure:
    """Growth of $1 in the portfolio vs the benchmark."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(portfolio_curve.index, portfolio_curve, label="Portfolio", linewidth=1.5)
    ax.plot(benchmark_curve.index, benchmark_curve, label="Benchmark", linewidth=1.5)
    ax.set_ylabel("Value of $1 invested")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def drawdown_figure(drawdown: pd.Series) -> Figure:
    """Portfolio drawdown through time, shaded below zero."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.4)
    ax.plot(drawdown.index, drawdown, linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def rolling_metric_figure(series: pd.Series, title: str, ylabel: str, as_percent: bool = False) -> Figure:
    """Generic rolling-metric line chart (rolling volatility and rolling Sharpe)."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series, linewidth=1.2)
    if as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def sharpe_histogram_figure(sharpes: np.ndarray, historical_sharpe: float) -> Figure:
    """Distribution of simulated Sharpe ratios with the historical value marked."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sharpes, bins=60, alpha=0.8)
    ax.axvline(
        historical_sharpe,
        color="crimson",
        linewidth=2,
        label=f"Historical Sharpe = {historical_sharpe:.2f}",
    )
    ax.set_xlabel("Simulated Sharpe ratio")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/plots.py tests/test_plots.py
git commit -m "feat: add matplotlib figure builders" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: app.py — the Streamlit dashboard

**Files:**
- Create: `app.py`

No pytest here (it's the UI shell — everything it calls is already tested); verification is running the app headless and loading the page.

- [ ] **Step 1: Write the app**

`app.py`:
```python
"""Streamlit dashboard. UI only — all math lives in src/ (see design spec)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src import benchmark, monte_carlo, plots, returns, risk
from src.data_loader import ValidationError, load_returns

DEFAULT_SAMPLE = Path(__file__).parent / "data" / "sample_real.csv"

EXPLAIN_SUMMARY = """
**Total return** compounds every daily return; **annualized return** is the
geometric (CAGR) rate that would compound to the same total over a 252-day year.
**Sharpe** is mean daily excess return over its standard deviation, annualized by
sqrt(252); **Sortino** replaces the denominator with downside deviation, penalizing
only returns below the target. **Volatility** is the annualized standard deviation of
daily returns. **Max drawdown** is the worst peak-to-trough loss of the equity curve.
**Beta** is the CAPM regression slope against the benchmark; **alpha** is its intercept
(annualized ×252) — the average return not explained by benchmark exposure.
"""

EXPLAIN_PERFORMANCE = """
The equity curve shows the growth of $1: V_t = V_{t-1}(1 + r_t). Returns
compound — the order-independent product, not the sum. Comparing the two lines
shows out/under-performance cumulatively, but says nothing yet about how much
risk was taken to achieve it.
"""

EXPLAIN_RISK_TIME = """
**Drawdown** measures the distance below the highest value reached so far —
realized peak-to-trough loss. It captures something volatility cannot: two
portfolios with identical volatility can have very different worst losses.
The **rolling** charts recompute volatility and Sharpe inside a moving window,
showing that risk and risk-adjusted performance are not constant. The first
window-minus-one days are blank on purpose — no partial windows.
"""

EXPLAIN_BENCHMARK = """
**Correlation** (−1 to 1) measures direction of co-movement; zero correlation
does not imply independence. **Beta** = Cov(portfolio, benchmark)/Var(benchmark)
is the regression slope: how much the portfolio moves per 1% benchmark move —
correlation rescaled by relative volatilities, so the two can tell different
stories. **Alpha** is the regression intercept: average return not explained by
market exposure — an estimate with sampling error, not proof of skill.
**R²** = correlation² is the share of portfolio variance the benchmark explains.
"""

EXPLAIN_MC = """
Treat history as ONE sample from an assumed return process: Normal(mu_hat,
sigma_hat²) fitted to the observed mean and standard deviation. Simulate 10,000
alternate histories of the same length, compute each one's Sharpe with the same
formula, and look at the spread: even with the process fixed, a finite sample
can produce very different measured Sharpes. Caveats: real returns are not
normal or independent, and the parameters are themselves estimates — so this
understates true uncertainty. It is a demonstration of sampling error, not a
forecast.
"""

st.set_page_config(page_title="Quantitative Risk & Performance Analyzer", layout="wide")
st.title("Quantitative Risk & Performance Analyzer")

# ---------- Sidebar ----------
st.sidebar.header("Data & Settings")
uploaded = st.sidebar.file_uploader(
    "Upload returns CSV",
    type="csv",
    help="Columns: date, portfolio_return, benchmark_return — daily returns as decimals (0.0021 = 0.21%)",
)
rf_percent = st.sidebar.number_input("Risk-free rate (annual, %)", value=0.0, step=0.25, format="%.2f")
window = st.sidebar.selectbox("Rolling window (days)", [30, 60, 90], index=1)
rf_annual = rf_percent / 100

source = uploaded if uploaded is not None else DEFAULT_SAMPLE
try:
    df, messages = load_returns(source)
except ValidationError as e:
    st.error(str(e))
    st.stop()

if uploaded is None:
    st.sidebar.caption("Using the bundled sample: AAPL/MSFT/NVDA/AMZN equal-weight vs SPY, 2022–2024.")
st.sidebar.caption(f"{len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
for m in messages:
    (st.sidebar.warning if m.severity == "warning" else st.sidebar.info)(m.text)

port, bench = df["portfolio"], df["benchmark"]


@st.cache_data
def cached_sharpe_distribution(portfolio: pd.Series, rf: float):
    return monte_carlo.simulate_sharpe_distribution(portfolio, rf_annual=rf)


def pct(x: float) -> str:
    return f"{x:.1%}"


def num(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.2f}"


# ---------- Summary metrics ----------
sharpe = risk.sharpe_ratio(port, rf_annual)
sortino = risk.sortino_ratio(port, rf_annual)
row1 = st.columns(4)
row1[0].metric("Total Return", pct(returns.cumulative_return(port)))
row1[1].metric("Annualized Return", pct(returns.annualized_return(port)))
row1[2].metric("Sharpe", num(sharpe))
row1[3].metric("Sortino", num(sortino))
row2 = st.columns(4)
row2[0].metric("Volatility (ann.)", pct(risk.annualized_volatility(port)))
row2[1].metric("Max Drawdown", pct(risk.max_drawdown(port)))
row2[2].metric("Beta", num(benchmark.beta(port, bench, rf_annual)))
row2[3].metric("Alpha (ann.)", pct(benchmark.alpha_annualized(port, bench, rf_annual)))
if pd.isna(sortino):
    st.caption("Sortino is undefined: no daily return fell below the target, so downside deviation is zero.")
with st.expander("What am I looking at?"):
    st.markdown(EXPLAIN_SUMMARY)

# ---------- Performance ----------
st.header("Portfolio Performance")
st.pyplot(plots.equity_curve_figure(returns.equity_curve(port), returns.equity_curve(bench)))
b_date, b_val = returns.best_day(port)
w_date, w_val = returns.worst_day(port)
c1, c2, c3 = st.columns(3)
c1.caption(f"Best day: {pct(b_val)} on {b_date.date()}")
c2.caption(f"Worst day: {pct(w_val)} on {w_date.date()}")
c3.caption(f"Mean daily return: {returns.mean_daily_return(port):.4%}")
with st.expander("What am I looking at?"):
    st.markdown(EXPLAIN_PERFORMANCE)

# ---------- Risk through time ----------
st.header("Risk Through Time")
st.pyplot(plots.drawdown_figure(risk.drawdown_series(port)))
if len(df) >= window:
    st.pyplot(
        plots.rolling_metric_figure(
            risk.rolling_volatility(port, window),
            f"Rolling {window}-day volatility (annualized)",
            "Volatility",
            as_percent=True,
        )
    )
    st.pyplot(
        plots.rolling_metric_figure(
            risk.rolling_sharpe(port, window, rf_annual),
            f"Rolling {window}-day Sharpe (annualized)",
            "Sharpe",
        )
    )
else:
    st.caption(f"Not enough data for a {window}-day rolling window ({len(df)} rows).")
with st.expander("What am I looking at?"):
    st.markdown(EXPLAIN_RISK_TIME)

# ---------- Benchmark analysis ----------
st.header("Benchmark Analysis")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Correlation", num(benchmark.correlation(port, bench)))
b2.metric("Beta", num(benchmark.beta(port, bench, rf_annual)))
b3.metric("Alpha (ann.)", pct(benchmark.alpha_annualized(port, bench, rf_annual)))
b4.metric("R²", num(benchmark.r_squared(port, bench)))
with st.expander("What am I looking at?"):
    st.markdown(EXPLAIN_BENCHMARK)

# ---------- Monte Carlo ----------
st.header("Monte Carlo Performance Uncertainty")
sharpes = cached_sharpe_distribution(port, rf_annual)
summary = monte_carlo.summarize_distribution(sharpes)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Historical Sharpe", num(sharpe))
m2.metric("Mean Simulated", num(summary["mean"]))
m3.metric("5th Percentile", num(summary["p5"]))
m4.metric("95th Percentile", num(summary["p95"]))
st.pyplot(plots.sharpe_histogram_figure(sharpes, sharpe))
st.caption(
    f"Median {num(summary['median'])}, std {num(summary['std'])} across "
    f"{len(sharpes):,} simulated histories of {len(port)} days each."
)
with st.expander("What am I looking at?"):
    st.markdown(EXPLAIN_MC)
```

- [ ] **Step 2: Verify the app boots and serves**

Run:
```bash
.venv/bin/streamlit run app.py --server.headless true --server.port 8501 &
APP_PID=$!
sleep 8
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
kill $APP_PID
```
Expected: `200`. Also confirm no traceback in the streamlit startup output. If available, use the app-verification tooling (e.g., a browser screenshot of localhost:8501) to confirm: two metric rows populated from the bundled real sample, five charts rendered, expanders present, sidebar shows "Using the bundled sample".

- [ ] **Step 3: Run the full suite (regression check)**

Run: `.venv/bin/pytest`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit dashboard with per-section explainers" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: MILESTONE PUSH — working dashboard**

```bash
git push
```

---

### Task 11: docs/concepts.md — the interview study guide

**Files:**
- Create: `docs/concepts.md`

- [ ] **Step 1: Write the study guide**

Write `docs/concepts.md` with the following structure and content. Each answer must state the convention exactly as implemented (this file is rehearsal material — first-person, direct answers, no hedging filler). Expand each bullet into 2–4 full sentences; the content below is the required substance:

```markdown
# Concepts — Interview Study Guide

Answers use the exact conventions implemented in src/ (see the design spec).

## Returns
- **What is a return?** The fractional change in value over one period: r_t = V_t/V_{t-1} − 1.
- **Why do returns compound?** Each day's return applies to the value produced by all previous days: V_n = V_0·∏(1+r_t). The product, not the sum — +10% then −10% loses 1%.
- **Average vs cumulative return:** the arithmetic mean describes a typical single day; the cumulative return describes the whole path. Volatility drags the compounded result below what the mean suggests.
- **How annualization works:** I report CAGR, (1+total)^(252/N) − 1 — the constant annual rate that compounds to the same total. The arithmetic alternative (mean×252) is larger whenever volatility > 0; Sharpe uses the arithmetic daily mean, which is why the two "annual returns" can disagree.

## Volatility
- **Variance** is the mean squared deviation from the mean — spread in squared units. **Standard deviation** is its square root, back in return units, which is why volatility is quoted as std.
- **Why ddof=1:** the sample mean is estimated from the same data, which removes one degree of freedom; dividing by N−1 makes the variance estimate unbiased. (np.std defaults to ddof=0; pandas to ddof=1 — my code pins ddof=1 everywhere.)
- **Why sqrt(252):** variances of independent returns add across days, so annual variance = 252×daily variance and std scales by √252. The independence assumption is the fine print.

## Sharpe
- **What it measures:** excess return per unit of total volatility: mean(r − rf_daily)/std(excess)·√252.
- **Why subtract the risk-free rate:** only return above the riskless alternative compensates risk-taking.
- **Why divide by volatility:** leverage can scale raw returns arbitrarily; return-per-unit-risk is leverage-invariant.
- **Limitations:** treats upside and downside symmetrically, assumes std captures risk, ignores autocorrelation and tails — and an observed Sharpe is a noisy estimate of the true one (the Monte Carlo section quantifies exactly this).

## Sortino
- **Difference from Sharpe:** the denominator is downside deviation — √(mean over ALL N days of min(r − target, 0)²) with target = rf_daily — so only below-target returns are penalized.
- **Why full-sample averaging:** dividing by only the count of down days (a common mistake) shrinks the denominator and inflates the ratio.
- **When it tells a different story:** for right-skewed strategies (big up days, small down days) Sortino ≫ Sharpe; for strategies with rare large losses it is the other way.

## Drawdown
- **Max drawdown** is the worst peak-to-trough loss: min of V_t/peak_t − 1, with the peak including the starting $1.
- **Why volatility isn't enough:** volatility measures typical dispersion, drawdown measures realized sequences of losses — same-volatility portfolios can differ wildly in worst loss.

## Correlation
- Normalized covariance, Cov/(σσ), bounded in [−1, 1] by Cauchy–Schwarz.
- **Zero correlation does not imply independence** — it rules out only linear association (y = x² on symmetric x has zero correlation and full dependence).

## CAPM
- **Beta** = Cov(excess_p, excess_b)/Var(excess_b): the regression slope; sensitivity to benchmark moves.
- **Beta vs correlation:** beta = ρ·(σ_p/σ_b) — correlation is direction only, beta includes scale.
- **Alpha** = intercept = mean(y) − β·mean(x), annualized ×252 for display: average return not explained by benchmark exposure. An estimate with sampling error — not proof of skill.
- **R²** = ρ²: the share of portfolio variance the benchmark explains.
- Subtracting a constant rf_daily leaves covariances unchanged, so beta/ρ/R² are identical on raw vs excess returns; only alpha moves.

## Monte Carlo
- **What it is:** repeatedly draw outcomes from an assumed probability model to see the distribution of a statistic.
- **Why I used it:** to show that the Sharpe I measured is one draw from a sampling distribution, not the true Sharpe.
- **Where the simulated returns come from:** Normal(μ̂, σ̂²) with parameters fitted to the historical sample; each simulated history has the same length N as the real one; same Sharpe formula applied.
- **Why simulations differ:** each history is a different finite random sample, so its mean and std — and hence Sharpe — differ.
- **Assumptions and why they're imperfect:** normality (real returns have fat tails and skew), independence (real returns cluster in volatility), and fixed plug-in parameters (μ̂, σ̂ are themselves uncertain) — all three make my interval an understatement of true uncertainty.
- **Sample size:** the standard error of the Sharpe estimate shrinks like 1/√N, so longer histories pin it down better.

## Dashboard
- **Why Streamlit:** a pure-Python way to publish the analysis; no frontend stack to maintain.
- **Data flow:** upload → data_loader validates → pure functions in src/ compute → app.py arranges → plots.py renders.
- **Why logic is separate from UI:** the finance code is testable (52 tests, including scipy cross-checks) and reusable without Streamlit; app.py contains zero math.
```

Adjust the test count in the last bullet to the real number from `.venv/bin/pytest` output.

- [ ] **Step 2: Cross-check against the dashboard expanders**

Read the five `EXPLAIN_*` strings in `app.py` and confirm nothing in concepts.md contradicts them (same conventions, same caveats). Fix any drift in concepts.md.

- [ ] **Step 3: Commit**

```bash
git add docs/concepts.md
git commit -m "docs: add interview study guide (concepts.md)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: README, final verification, milestone push

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
```markdown
# Quantitative Risk & Performance Analyzer

A small, fully hand-rolled quantitative finance tool: upload a CSV of daily
portfolio and benchmark returns and get performance, risk, benchmark-exposure,
and Monte Carlo Sharpe-uncertainty analysis in a Streamlit dashboard.

Every formula (CAGR, volatility, Sharpe, Sortino, drawdown, CAPM beta/alpha/R²,
the Monte Carlo simulation) is implemented explicitly in numpy/pandas —
scipy appears only in the test suite as an independent cross-check.

## Run it

```bash
uv venv .venv
uv pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The dashboard opens preloaded with the bundled real sample
(AAPL/MSFT/NVDA/AMZN equal-weight vs SPY, 2022–2024). Upload your own CSV to
replace it.

## Input format

| column | meaning |
|---|---|
| `date` | one trading day per row |
| `portfolio_return` | daily return as a decimal (0.0021 = 0.21%) |
| `benchmark_return` | daily benchmark return as a decimal |

At least 60 rows. The loader sorts unordered rows, drops incomplete rows with
a warning, warns when values look like percentages, and rejects duplicate or
unparseable dates.

## Tests

```bash
uv pip install -r requirements-dev.txt
.venv/bin/pytest
```

Three layers: hand-computed known answers, scipy/pandas referee cross-checks,
and behavioral/edge tests (including an end-to-end check that the pipeline
recovers the synthetic sample's known beta and alpha).

## Sample data provenance

- `data/sample_synthetic.csv` — generated by `data/generate_synthetic.py`
  (seeded; known ground truth: beta 1.3, alpha 2%/yr, R² 0.75).
- `data/sample_real.csv` — produced once by `data/fetch_real.py` (yfinance,
  not a dependency) and committed as a static file.

## Project layout

```
app.py            Streamlit UI (no math)
src/              pure analysis modules: data_loader, returns, risk,
                  benchmark, monte_carlo, plots
tests/            pytest suite
data/             bundled samples + provenance scripts
docs/concepts.md  interview study guide for every formula used
```

## Conventions

Returns are decimals; every std is sample std (ddof=1); annualization uses 252
trading days; risk-free rate converts as rf/252; ratios with zero denominators
report NaN, never inf. Full details: `docs/superpowers/specs/`.

## Intentionally out of scope

Strategy backtesting, live data, portfolio optimization, factor models, GARCH,
VaR systems — this project is deliberately small; see the design spec.
```

- [ ] **Step 2: Full verification**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (≈52). Then boot the app once more as in Task 10 Step 2 and confirm `200`.

- [ ] **Step 3: Commit and MILESTONE PUSH — project complete**

```bash
git add README.md
git commit -m "docs: add README" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

## Spec coverage self-check (for the reviewer)

- Spec §3 architecture/purity → Tasks 1, 2–9 module boundaries, app.py in Task 10
- Spec §4 conventions → constants (T1), returns (T2), risk (T3–4), benchmark (T5), MC (T6), formatting helpers `pct`/`num` (T10)
- Spec §5 validation table → Task 7 (one test per rule); N<window caption → Task 10
- Spec §6 sample data → Task 8 (generator, fetch, auto-load default wired in Task 10)
- Spec §7 test layers → known answers (T2–T4), referees (T5), behavioral/edge (T3–T8), determinism (T6), e2e recovery (T8)
- Spec §8 dashboard order/sidebar/expanders/cache → Task 10
- Spec §9 concepts.md → Task 11
- Spec §12 definition of done → Tasks 10 (1–3), 11 (4), 8+12 (5), 12 (6)
