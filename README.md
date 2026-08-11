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

| | Section | What it does and why |
|---|---|---|
| 1 | **Summary metrics** | The two metric rows summarize the whole analysis at a glance: total return, annualized (CAGR) return, Sharpe, Sortino, annualized volatility, maximum drawdown, beta, and alpha, all computed live from the loaded returns. The Sortino ratio uses full-sample downside deviation rather than the down-days-only variant that inflates the ratio. |
| 2 | **Portfolio performance** | The equity curve shows whether the portfolio made money and how that compares to the benchmark. It compounds daily returns into the growth of $1 (a product of returns, not a sum) and plots both series from the same starting dollar so the comparison is fair at any scale. The best and worst single days are called out below the chart. |
| 3 | **Risk through time** | This section shows what the ride felt like rather than just its average. The drawdown chart measures losses against a running peak that includes the starting dollar, so a first-day loss already counts. Rolling volatility and rolling Sharpe recompute risk inside a user-selectable 30/60/90-day window, using strict full windows only, so partial-window artifacts never appear. |
| 4 | **Benchmark analysis** | The CAPM regression answers whether performance came from skill or from market exposure. Correlation, beta, R², and alpha are computed as closed-form covariance algebra written out by hand, with no regression library in the source code. The test suite proves the results match `scipy.stats.linregress` to a tolerance of 1e-12. |
| 5 | **Monte Carlo** | This section measures how much the headline Sharpe ratio can be trusted. It fits a normal distribution to the observed returns, simulates 10,000 histories of the same length, and scores each one with the exact same Sharpe function used for the headline metric. On the bundled three-year sample the simulated Sharpes have a standard deviation of about 0.58, which quantifies how noisy the estimate is. |

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
