# Quantitative Risk & Performance Analyzer

A fully hand-rolled quantitative finance tool: upload a CSV of daily portfolio
and benchmark returns and get a complete risk and performance report — how the
portfolio grew, how much risk it took along the way, how much of the result is
explained by market exposure, and how much confidence the headline numbers
actually deserve — in an interactive Streamlit dashboard.

Every formula (CAGR, volatility, Sharpe, Sortino, drawdown, CAPM beta/alpha/R²,
the Monte Carlo simulation) is implemented explicitly in numpy/pandas rather
than imported from a finance library — scipy appears only in the test suite as
an independent cross-check. Each dashboard section carries an expander
explaining the math it displays, and the app opens preloaded with a real
three-year sample (an AAPL/MSFT/NVDA/AMZN equal-weight portfolio against SPY,
2022–2024), so everything in the screenshots below is live output.

**[▶ Try the live demo](https://rfukuda06-quant-risk-app-r2epvs.streamlit.app/)** — no setup required.

![Dashboard overview — summary metrics and equity curve](docs/screenshots/dashboard-overview.png)

![Risk through time — drawdown and rolling metrics](docs/screenshots/risk-through-time.png)

![Benchmark analysis and Monte Carlo Sharpe uncertainty](docs/screenshots/benchmark-monte-carlo.png)

## Key components

The dashboard is a chain of questions — each section answers something the
previous one cannot:

| | Section | Question it answers | What's computed |
|---|---|---|---|
| 1 | **Summary metrics** | The headlines, at a glance | Total & annualized (CAGR) return, Sharpe, Sortino, volatility, max drawdown, beta, alpha |
| 2 | **Portfolio performance** | Did it make money? | Equity curve — growth of $1, portfolio vs benchmark — plus best/worst day |
| 3 | **Risk through time** | What did the ride look like? | Drawdown (realized peak-to-trough loss), rolling volatility, rolling Sharpe |
| 4 | **Benchmark analysis** | Skill, or just market exposure? | Hand-rolled CAPM: correlation, beta, R², and alpha (the return the market can't explain) |
| 5 | **Monte Carlo** | How much should you trust these numbers? | 10,000 simulated same-length histories → the sampling distribution of the Sharpe ratio |

Upstream of all five, a validation layer vets every CSV — required columns,
unique parseable dates, decimal-scale values, ≥ 60 rows — before any metric
is computed.

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
