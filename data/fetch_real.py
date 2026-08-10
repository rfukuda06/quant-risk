"""One-time script that produced data/sample_real.csv. yfinance is NOT a
project dependency; to re-run: uv run --with yfinance --no-project python data/fetch_real.py

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
