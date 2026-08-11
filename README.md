# Quantitative Risk & Performance Analyzer

A hand-rolled quantitative finance tool: upload a CSV of daily portfolio and
benchmark returns and get an interactive Streamlit dashboard covering
performance, risk, benchmark exposure, and the statistical uncertainty behind
the headline numbers. Every formula is implemented from scratch in
numpy/pandas and cross-checked against independent references in the test
suite, and the app opens preloaded with a real three-year sample
(AAPL/MSFT/NVDA/AMZN equal-weight vs SPY, 2022–2024).

**[▶ Try the live demo](https://rfukuda06-quant-risk-app-r2epvs.streamlit.app/)** — no setup required.

![Dashboard overview — summary metrics and equity curve](docs/screenshots/dashboard-overview.png)

![Risk through time — drawdown and rolling metrics](docs/screenshots/risk-through-time.png)

![Benchmark analysis and Monte Carlo Sharpe uncertainty](docs/screenshots/benchmark-monte-carlo.png)

## Key components

| | Section | What it does — and how it's built |
|---|---|---|
| 1 | **Summary metrics** | Eight headline stats computed live from the loaded returns: total and CAGR return, Sharpe, Sortino, annualized volatility, max drawdown, CAPM beta and alpha. Sortino uses full-sample downside deviation (not the down-days-only shortcut that inflates it), and degenerate inputs render as "—" instead of crashing or showing inf |
| 2 | **Portfolio performance** | Compounds daily returns into the growth of $1 — the order-independent product, not the sum — for portfolio and benchmark on one axis, with best/worst-day callouts. Normalizing both series to $1 makes the comparison leverage- and scale-fair |
| 3 | **Risk through time** | Drawdown measured against a running peak that includes the starting dollar, so a first-day loss already counts; rolling volatility and Sharpe over a user-selectable 30/60/90-day window, with strict full windows only — the first window−1 days are deliberately blank rather than showing partial-window artifacts |
| 4 | **Benchmark analysis** | CAPM as closed-form covariance algebra written out by hand — slope = Cov/Var, intercept, correlation, R² — no regression library in the source. The test suite proves it matches `scipy.stats.linregress` to 1e-12 |
| 5 | **Monte Carlo** | Fits Normal(μ̂, σ̂²) to the observed returns, simulates 10,000 same-length histories, and scores each with the exact same Sharpe function as the headline metric. Seeded for reproducibility and cached for responsiveness; on the bundled 3-year sample the simulated Sharpes spread with std ≈ 0.58 — the headline Sharpe is a noisy estimate, and this section shows by how much |

The order is deliberate: returns first, then the risk taken to earn them, then
how much of the result the benchmark explains, then how much sampling noise
sits in the estimates. Every metric is written from scratch, tested against
independent references, and displayed with its assumptions stated. A
validation layer vets every CSV (required columns, unique parseable dates,
decimal-scale values, ≥ 60 rows) before any metric runs.

## Tests

```bash
uv pip install -r requirements-dev.txt
.venv/bin/pytest
```

The 54-test suite works in three layers:

- **Hand-computed known answers** — every formula is checked against small
  worked examples (+10% then −10% must equal exactly −1%), so a wrong sign,
  a ddof slip, or a bad annualization fails immediately.
- **Independent referees** — the hand-rolled CAPM and correlation are compared
  against `scipy.stats.linregress` and `np.corrcoef` at 1e-12 tolerance; the
  math must agree with libraries the source code never imports.
- **Behavioral and end-to-end** — edge cases (zero volatility → NaN, never
  inf; malformed CSVs → clean user-facing errors; seeded Monte Carlo
  determinism) plus a ground-truth recovery test: the pipeline must
  re-estimate the synthetic generator's known beta 1.3, alpha 2%/yr, and
  R² 0.75 from its own output.

## Input format

The app works out of the box with a bundled default dataset
(AAPL/MSFT/NVDA/AMZN equal-weight vs SPY, 2022–2024); upload your own CSV in
the sidebar to analyze your data instead. The file needs:

| column | meaning |
|---|---|
| `date` | one trading day per row |
| `portfolio_return` | daily return as a decimal (0.0021 = 0.21%) |
| `benchmark_return` | daily benchmark return as a decimal |

At least 60 rows. The loader sorts unordered rows, drops incomplete rows with
a warning, warns when values look like percentages, and rejects duplicate or
unparseable dates.

## Run it locally

The [live demo](https://rfukuda06-quant-risk-app-r2epvs.streamlit.app/) above
needs no setup. To run the app on your own machine instead:

```bash
uv venv .venv
uv pip install -r requirements.txt
.venv/bin/streamlit run app.py
```
