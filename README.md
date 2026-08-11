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

| | Section | What it shows |
|---|---|---|
| 1 | **Summary metrics** | Total & annualized (CAGR) return, Sharpe, Sortino, volatility, max drawdown, beta, alpha — the whole report at a glance before the sections below unpack it |
| 2 | **Portfolio performance** | The equity curve compounds daily returns into the growth of $1, portfolio vs benchmark, with best/worst-day callouts — did it make money, and against what? |
| 3 | **Risk through time** | Drawdown tracks realized peak-to-trough losses — the pain volatility alone can't show — while rolling volatility and rolling Sharpe reveal that risk and risk-adjusted performance change through time |
| 4 | **Benchmark analysis** | A hand-rolled CAPM regression splits returns into market-driven and unexplained parts (correlation, beta, R², alpha) — was the performance skill, or just market exposure? |
| 5 | **Monte Carlo** | Simulates 10,000 alternate same-length return histories to build the sampling distribution of the Sharpe ratio — how different the measured Sharpe could have looked by luck alone |

The order is deliberate: each section answers the question the previous one
leaves open. A return chart can't say what the ride cost, so risk metrics
follow; risk can't say whether returns were earned or simply borrowed from
market exposure, so the CAPM decomposition follows; and even alpha and Sharpe
are point estimates on one finite sample, so the dashboard closes by measuring
how much sampling luck alone could move them. That last step is the project's
purpose in miniature — not just computing standard portfolio metrics, but
hand-building each one, knowing exactly what it assumes, and being honest
about how much confidence the numbers deserve. Upstream of it all, a
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
