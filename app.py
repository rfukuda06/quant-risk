"""Streamlit dashboard. UI only — all math lives in src/ (see design spec)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src import benchmark, monte_carlo, plots, returns, risk
from src.data_loader import ValidationError, load_returns

DEFAULT_SAMPLE = Path(__file__).parent / "data" / "sample_real.csv"

EXPLAIN_SUMMARY = """
- **Total return** compounds every daily return.
- **Annualized return** is the geometric (CAGR) rate that would compound to the same total over a 252-day year.
- **Sharpe** is mean daily excess return over its standard deviation, annualized by √252.
- **Sortino** replaces the denominator with downside deviation, penalizing only returns below the target.
- **Volatility** is the annualized standard deviation of daily returns.
- **Max drawdown** is the worst peak-to-trough loss of the equity curve.
- **Beta** is the CAPM regression slope against the benchmark.
- **Alpha** is the CAPM regression intercept (annualized ×252) — the average return not explained by benchmark exposure.
"""

EXPLAIN_PERFORMANCE = """
The equity curve shows the growth of $1: Vₜ = Vₜ₋₁(1 + rₜ). Returns
compound — the order-independent product, not the sum. Comparing the two lines
shows out/under-performance cumulatively, but says nothing yet about how much
risk was taken to achieve it.
"""

EXPLAIN_RISK_TIME = """
- **Drawdown** measures the distance below the highest value reached so far — realized peak-to-trough loss. It captures something volatility cannot: two portfolios with identical volatility can have very different worst losses.
- The **rolling** charts recompute volatility and Sharpe inside a moving window, showing that risk and risk-adjusted performance are not constant. The first window-minus-one days are blank on purpose — no partial windows.
"""

EXPLAIN_BENCHMARK = """
- **Correlation** (−1 to 1) measures direction of co-movement; zero correlation does not imply independence.
- **Beta** = Cov(portfolio, benchmark)/Var(benchmark) is the regression slope: how much the portfolio moves per 1% benchmark move — correlation rescaled by relative volatilities, so the two can tell different stories.
- **Alpha** is the regression intercept: average return not explained by market exposure — an estimate with sampling error, not proof of skill.
- **R²** = correlation² is the share of portfolio variance the benchmark explains.
"""

EXPLAIN_MC = """
Treat history as ONE sample from an assumed return process: Normal(μ̂, σ̂²)
fitted to the observed mean and standard deviation. Simulate 10,000
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

st.markdown("#### Data")
if uploaded is None:
    st.caption(
        "Type: default bundled sample  \n"
        "AAPL/MSFT/NVDA/AMZN equal-weight portfolio vs SPY benchmark  \n"
        f"{len(df)} rows  \n"
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )
else:
    st.caption(
        "Type: uploaded CSV  \n"
        f"`{uploaded.name}` — `portfolio_return` vs `benchmark_return`  \n"
        f"{len(df)} rows  \n"
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )

port, bench = df["portfolio"], df["benchmark"]


@st.cache_data
def cached_sharpe_distribution(portfolio: pd.Series, rf: float):
    return monte_carlo.simulate_sharpe_distribution(portfolio, rf_annual=rf)


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


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
with st.expander("Explanation"):
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
with st.expander("Explanation"):
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
with st.expander("Explanation"):
    st.markdown(EXPLAIN_RISK_TIME)

# ---------- Benchmark analysis ----------
st.header("Benchmark Analysis")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Correlation", num(benchmark.correlation(port, bench)))
b2.metric("Beta", num(benchmark.beta(port, bench, rf_annual)))
b3.metric("Alpha (ann.)", pct(benchmark.alpha_annualized(port, bench, rf_annual)))
b4.metric("R²", num(benchmark.r_squared(port, bench)))
with st.expander("Explanation"):
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
with st.expander("Explanation"):
    st.markdown(EXPLAIN_MC)
