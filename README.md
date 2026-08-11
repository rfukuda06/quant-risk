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
| 1 | **Summary metrics** | Two rows of headline statistics summarize the full analysis: total return, annualized (CAGR) return, Sharpe ratio, Sortino ratio, annualized volatility, maximum drawdown, beta, and alpha, all computed directly from the loaded return series. The Sortino ratio uses full-sample downside deviation rather than the down-days-only variant, which understates the denominator and inflates the ratio. |
| 2 | **Portfolio performance** | The equity curve measures cumulative performance relative to the benchmark. Daily returns are compounded multiplicatively into the value of $1 invested (Vₜ = Vₜ₋₁(1 + rₜ)), and both series are normalized to the same initial value so relative performance is comparable across scales. The largest single-day gain and loss are reported below the chart. |
| 3 | **Risk through time** | This section characterizes the time profile of risk rather than a single summary number. Drawdown is computed against a running maximum that includes the initial value, so a loss on the first observation registers immediately. Rolling volatility and rolling Sharpe are recomputed over a user-selectable 30/60/90-day window using complete windows only, which eliminates partial-window artifacts. |
| 4 | **Benchmark analysis** | A CAPM regression decomposes portfolio returns into benchmark-driven and residual components. Correlation, beta, R², and alpha are derived in closed form from hand-written covariance expressions, with no regression library in the source code; the test suite verifies agreement with `scipy.stats.linregress` to a tolerance of 1e-12. |
| 5 | **Monte Carlo** | This section quantifies the sampling uncertainty of the Sharpe ratio estimate. A normal distribution is fit to the observed returns, 10,000 histories of equal length are simulated, and each is scored with the same Sharpe implementation used for the headline metric. On the bundled three-year sample the simulated Sharpe ratios have a standard deviation of approximately 0.58. |

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
