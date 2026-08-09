# Quantitative Risk & Performance Analyzer — Design Spec

Date: 2026-08-09
Status: approved pending final user review

## 1. Purpose

A small, fully-explainable Python quantitative finance tool. It ingests a CSV of
daily portfolio and benchmark returns and presents performance, risk, benchmark
exposure, and performance-uncertainty analysis in a single-page Streamlit
dashboard. Primary goals: reinforce probability/statistics for quant trading
interviews, serve as a resume project, and be explainable line-by-line by the
author. Scope is intentionally strict — no features beyond this spec.

## 2. Decisions log (from brainstorming)

- Sample data: **synthetic + real** — a seeded synthetic generator (doubles as a
  test fixture) plus one real dataset fetched once and committed as a static CSV.
- Explanations live in **both** places: `st.expander` blocks under each dashboard
  section and a `docs/concepts.md` study guide.
- Implementation approach: **hand-rolled, strict scope** — all finance/stats
  formulas explicit in numpy/pandas; scipy appears only in the test suite as an
  independent cross-check. No extra metrics beyond the outline (no alpha t-stat,
  no analytic Sharpe standard-error cross-check).
- Rolling windows: selectable **30/60/90**, default 60.
- Monte Carlo simulates **only histories of the historical sample length N**.
  The sample-size experiment (outline §18) is removed entirely from
  implementation and dashboard.
- Dashboard auto-loads the bundled real sample on first open; uploading replaces it.

## 3. Architecture

```
quant-risk/
├── app.py                    # Streamlit UI only — zero math in this file
├── requirements.txt          # runtime: streamlit, pandas, numpy, matplotlib
├── requirements-dev.txt      # pytest, scipy
├── README.md
├── data/
│   ├── sample_synthetic.csv
│   ├── sample_real.csv       # committed static file; provenance: fetch_real.py
│   ├── generate_synthetic.py # seeded generator — reproduces sample_synthetic.csv
│   └── fetch_real.py         # one-time yfinance script (yfinance not in requirements)
├── docs/
│   └── concepts.md           # study guide answering the outline's §26 questions
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # CSV parsing + validation
│   ├── returns.py            # cumulative return, CAGR, equity curve, best/worst day
│   ├── risk.py               # volatility, Sharpe, Sortino, drawdown, rolling metrics
│   ├── benchmark.py          # correlation, beta, alpha, R²
│   ├── monte_carlo.py        # parameter estimation, simulation, distribution summary
│   └── plots.py              # all matplotlib figure-builders (return Figure objects)
└── tests/
    ├── test_data_loader.py
    ├── test_returns.py
    ├── test_risk.py
    ├── test_benchmark.py
    └── test_monte_carlo.py
```

**Purity rule:** no file in `src/` imports Streamlit. Every analysis function is
pure — takes a Series/DataFrame plus scalar parameters, returns scalars, Series,
or a small result object. `plots.py` functions take precomputed data and return
`matplotlib.figure.Figure`.

**Data flow:** upload (or bundled sample) → `data_loader.load_returns` validates
and returns a clean DataFrame (ascending `DatetimeIndex`; float64 columns
`portfolio`, `benchmark`) plus warnings → `app.py` calls the analysis modules,
arranges results into the dashboard layout (§8 below), and renders figures with
`st.pyplot`. The Monte Carlo call is wrapped in `st.cache_data`.

## 4. Quantitative conventions (pinned)

Global rules:

- Returns are decimals (0.01 = 1%). No automatic percent conversion, ever.
- Every standard deviation is the **sample** std, `ddof=1` (note: `np.std`
  defaults to `ddof=0`; pandas defaults to `ddof=1` — pinned and tested).
- Annualization factor: `TRADING_DAYS = 252`.
- Risk-free rate: annual input `rf` (default 0), converted as
  `rf_daily = rf / 252` (simple convention; the geometric alternative is noted
  in concepts.md as negligibly different). Daily excess series `r − rf_daily`
  feeds Sharpe, Sortino, and CAPM.

### returns.py

- Cumulative return: `prod(1 + r_t) − 1`.
- Equity curve: `V_t = prod_{s<=t}(1 + r_s)` from `V_0 = 1`; first plotted value
  is `1 + r_1`, final value equals `1 + cumulative`. Computed for both columns.
- Annualized return (headline): **CAGR** `= (1 + cumulative)^(252/N) − 1`.
- Mean daily return (arithmetic), best day and worst day (value and date).

### risk.py

- Volatility: daily `std(r, ddof=1)`; annualized `× sqrt(252)`.
- Sharpe (annualized): `mean(excess) / std(excess, ddof=1) × sqrt(252)`.
- Sortino (annualized): `mean(excess) / downside_dev × sqrt(252)` with
  `downside_dev = sqrt( mean( min(r − rf_daily, 0)² ) )` averaged over **all** N
  observations (full-sample definition; the negative-days-only variant is
  rejected). Target = `rf_daily`. If no observation falls below target:
  return `NaN` (rendered as "—" with a caption), never `inf`.
- Drawdown: `P_t = cummax(V_t)`, `D_t = V_t / P_t − 1` (≤ 0 everywhere);
  max drawdown `= min(D_t)`; current drawdown = last value.
- Rolling metrics: window `w ∈ {30, 60, 90}`, default 60, `min_periods = w`
  (first `w−1` points are NaN gaps — no partial windows). Rolling volatility and
  rolling Sharpe reuse the definitions above per window. Windows with zero
  rolling std yield `NaN`.

### benchmark.py

- CAPM fitted on excess returns via closed-form simple-OLS estimators:
  `beta = Cov(x, y, ddof=1) / Var(x, ddof=1)`,
  `alpha_daily = mean(y) − beta · mean(x)` where `y = r_p − rf_daily`,
  `x = r_m − rf_daily`.
- Displayed alpha is annualized: `alpha_annual = alpha_daily × 252`.
- Correlation: Pearson. `R² = ρ²` (simple-regression identity, tested).
  Note recorded in concepts.md: subtracting a constant `rf_daily` leaves
  covariances unchanged, so beta, ρ, and R² are identical on raw vs excess
  returns; only alpha moves.

### monte_carlo.py

- Estimate `mu_hat`, `sigma_hat` (ddof=1) from historical portfolio returns.
- Simulate one array of shape `(10_000, N)` from `Normal(mu_hat, sigma_hat²)`
  using `np.random.default_rng(seed=42)`. `N` is **always the historical sample
  length** — no other lengths, no sample-size experiment.
- `n_sims = 10_000` and `seed = 42` are documented module constants, not UI
  controls.
- Each simulated row's Sharpe is computed by the **same `risk.py` Sharpe
  function** with the same `rf`, so histogram and historical marker are
  apples-to-apples.
- Summary: mean, median, std, 5th percentile, 95th percentile.

### Display formatting

- Percentages, 1 decimal: returns, volatility, drawdown, alpha.
- Plain numbers, 2 decimals: Sharpe, Sortino, beta, correlation, R².

## 5. Validation rules (data_loader.py)

`load_returns(file_or_buffer) -> (DataFrame, list[LoaderMessage])`; hard
failures raise `ValidationError` with a user-readable message (`app.py` renders
it via `st.error` and stops). `LoaderMessage` carries severity (`info`/`warning`)
and text.

| Condition | Behavior |
|---|---|
| Missing required column (`date`, `portfolio_return`, `benchmark_return`; case-insensitive; extra columns ignored with info note) | **Error** |
| Unparseable dates or non-numeric returns | **Error**, citing first offending rows |
| Duplicate dates | **Error** (ambiguous data) |
| Unsorted dates | Sort ascending, **info** note |
| Rows with missing values | Drop, **warning** with count |
| Median absolute return > 0.05 | **Warning**: "values look like percentages, not decimals" (no auto-convert) |
| Any single absolute return > 0.5 | **Warning** listing dates; analysis proceeds |
| Fewer than 60 rows (after drops) | **Error** |
| N < selected rolling window | Rolling section shows a caption instead of charts (handled in `app.py`) |

Output frame: ascending `DatetimeIndex`, float64 columns `portfolio`, `benchmark`.

## 6. Sample data

**Synthetic** (`data/generate_synthetic.py`, fixed seed 7, 756 business days
ending 2025-12-31):

- Benchmark: iid `Normal(0.08/252, (0.16/sqrt(252))²)` — 8% annual drift, 16%
  annual vol.
- Portfolio: `alpha_daily + 1.3 × benchmark_t + eps_t` with
  `alpha_daily = 0.02/252` (2% annual alpha), `beta_true = 1.3`, and noise
  `eps ~ Normal(0, sigma_eps²)` sized for `R² ≈ 0.75` via
  `sigma_eps = beta_true · sigma_m · sqrt((1 − R²)/R²)`.
- Ground truth documented in the script header; an end-to-end test asserts the
  app's estimates recover it within tolerance.

**Real** (`data/fetch_real.py`, run once, output committed):

- Portfolio: equal-weight daily mean of AAPL, MSFT, NVDA, AMZN adjusted-close
  daily returns (daily-rebalanced equal weighting).
- Benchmark: SPY. Period: 2022-01-01 → 2024-12-31 (includes the 2022 drawdown
  and 2023–24 rally; beta visibly ≠ 1).
- yfinance is required only to re-run the script; it is not a project dependency.

`sample_real.csv` is the dashboard's auto-loaded default.

## 7. Testing plan (pytest)

1. **Hand-computed known answers:** `[0.10, −0.05]` → cumulative exactly 0.045;
   constant daily return → CAGR exactly `(1+r)^252 − 1`; crafted curve rising to
   2 then falling to 1 → max drawdown exactly −0.5; a 4-value Sortino
   downside-deviation case worked by hand; zero-vol and no-down-days cases →
   `NaN`, never `inf`.
2. **Referee tests (scipy/pandas as independent implementations):** beta, alpha,
   R² vs `scipy.stats.linregress`; correlation vs `np.corrcoef`; volatility and
   Sharpe vs an independent recomputation using pandas `.mean()`/`.std(ddof=1)`.
   Agreement to ~1e-12.
3. **Behavioral / edge:** drawdown ≤ 0 everywhere and = 0 at running peaks;
   every loader rule in §5 has a test (each error and each warning); Monte Carlo
   determinism under the fixed seed (identical summary on rerun), output shape,
   and mean simulated Sharpe near the plug-in Sharpe (loose tolerance);
   end-to-end synthetic fixture recovers `beta ≈ 1.3` and `alpha ≈ 2%` within
   tolerance.

## 8. Dashboard composition (app.py)

Sidebar: file uploader; risk-free-rate input in percent (default 0.0); rolling
window selectbox (30/60/90, default 60); caption when the bundled sample is in
use; data summary (row count, date range) and loader messages.

Main column, in order (mirrors outline §23):

1. Title.
2. Summary metrics — two `st.columns(4)` rows:
   Total Return / Annualized Return / Sharpe / Sortino, then
   Volatility / Max Drawdown / Beta / Alpha.
3. Performance — portfolio-vs-benchmark equity curve; captions for best day,
   worst day, mean daily return.
4. Risk through time — drawdown chart, rolling volatility, rolling Sharpe.
5. Benchmark analysis — Correlation / Beta / Alpha / R² metric row.
6. Monte Carlo — Historical Sharpe / Mean simulated / 5th pct / 95th pct metric
   row; histogram of simulated Sharpes with a vertical line at the historical
   Sharpe.

Each section (2–6) closes with `st.expander("What am I looking at?")` containing
a one-paragraph explanation distilled from the matching `concepts.md` entry.

## 9. concepts.md

Mirrors the outline's §26 headings — Returns, Volatility, Sharpe, Sortino,
Drawdown, Correlation, CAPM, Monte Carlo, Dashboard — as direct Q&A answering
the questions listed there, using the exact conventions of §4, including: why
CAGR ≠ mean×252, why `ddof=1`, why `sqrt(252)` assumes independence, why
full-sample downside deviation, why beta ≠ correlation, and why the Monte Carlo
understates true uncertainty (fixed plug-in parameters, normality, iid).

## 10. Environment & tooling

- Python 3.13, uv-managed virtualenv.
- `requirements.txt` (runtime): streamlit, pandas, numpy, matplotlib.
- `requirements-dev.txt`: pytest, scipy.
- Git repository on `main`; spec and plan documents live under `docs/`.

## 11. Non-goals

Everything in outline §20, plus, decided during design:

- Sample-size Monte Carlo experiment (outline §18) — removed entirely; the MC
  runs at the historical N only.
- Benchmark scatter plot with fitted regression line (outline lists it optional).
- Alpha t-statistic and analytic Sharpe standard-error cross-check.
- UI controls for simulation count or seed.
- Automatic percent→decimal conversion of inputs.

## 12. Definition of done

Maps to outline §28, amended by design decisions:

1. CSV upload through Streamlit with the §5 validation behavior, plus bundled
   samples (auto-loaded real sample by default).
2. All §4 metrics computed by hand-rolled numpy/pandas code in pure modules.
3. All §8 dashboard sections render with the §4 formatting and per-section
   explainer expanders.
4. `docs/concepts.md` complete per §9.
5. Full §7 test suite passing.
6. README covering setup (uv), running the app, data format, and sample-data
   provenance.
