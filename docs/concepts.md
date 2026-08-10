# Concepts — Interview Study Guide

Answers use the exact conventions implemented in src/ (see the design spec).

## Returns

- **What is a return?** The fractional change in value over one period: r_t = V_t/V_{t-1} − 1. It is dimensionless and represents the proportional gain or loss in a single day. This is the raw input the dashboard expects — daily returns as decimals, not prices.

- **Why do returns compound?** Each day's return applies to the value produced by all previous days: V_n = V_0·∏(1+r_t). The product, not the sum — +10% then −10% leaves you at 0.99, a 1% loss. This multiplicative structure is why the equity curve in the dashboard is computed as a running product, not a cumulative sum.

- **Average vs cumulative return:** The arithmetic mean describes a typical single day; the cumulative return describes the whole path as the total fractional change from start to finish. Volatility drags the compounded result below what the mean suggests — a phenomenon called volatility drag — which is why a high-mean strategy can still underperform a lower-mean, lower-volatility one over a long horizon.

- **How annualization works:** I report CAGR, (1+total)^(252/N) − 1 — the constant annual rate that compounds to the same total return over 252 trading days. At the daily level, the arithmetic mean always exceeds the geometric daily growth rate whenever volatility > 0 (AM–GM inequality); the gap is the volatility drag, ≈ σ²/2 per day. On the bundled real sample this translates to mean×252 ≈ 27.1% versus a CAGR of ≈ 24.8%. Sharpe uses the arithmetic daily mean scaled by √252, which is why the two "annual returns" shown on the dashboard can disagree and are labeled differently.

## Volatility

- **Variance** is the mean squared deviation from the mean — spread in squared units. **Standard deviation** is its square root, back in return units, which is why volatility is quoted as std rather than variance. Quoting variance would give numbers like 0.0001 where std gives 0.01 (1%), which is more interpretable.

- **Why ddof=1:** The sample mean is estimated from the same data, which uses up one degree of freedom; dividing by N−1 makes the variance estimate unbiased. `np.std` defaults to ddof=0; pandas defaults to ddof=1 — my code pins ddof=1 everywhere in both `np.asarray(...).std(ddof=1)` calls and `.rolling(...).std(ddof=1)` calls to ensure consistency.

- **Why sqrt(252):** Variances of independent daily returns add across days, so annual variance = 252×daily variance and standard deviation scales by √252. The independence assumption is the fine print — real returns exhibit volatility clustering, so this is an approximation that understates true risk in trending markets.

- **Why compare std to a tolerance instead of zero:** A constant return series should have std exactly 0, but floating-point summation leaves noise around 1e-19, so dividing by it would report a Sharpe near 1e17 instead of "undefined." I define `ZERO_TOL = 1e-10` in `src/constants.py` and treat any std below that threshold as zero, returning NaN. In `src/benchmark.py`, `beta` compares the benchmark's variance (a `_covariance(x, x)` call) against `ZERO_TOL**2`; `correlation` compares the product of the two standard deviations against `ZERO_TOL**2`. Real daily-return stds sit many orders of magnitude above the cutoff, so the guard never fires on genuine data.

## Sharpe

- **What it measures:** Excess return per unit of total volatility: mean(r − rf_daily) / std(excess) · √252. It answers: how much annualized return did I earn above the risk-free rate for each unit of total risk I bore? Higher is better, and a ratio above ~1 is generally considered good.

- **Why subtract the risk-free rate:** Only return above the riskless alternative compensates risk-taking. A government T-bill earns a return with zero equity risk, so the relevant question is how much the strategy earned *above* that baseline, not in absolute terms.

- **Why divide by volatility:** Leverage can scale raw returns arbitrarily — simply borrowing money amplifies both return and risk proportionally. Return-per-unit-risk is leverage-invariant: doubling the position doubles both numerator and denominator, leaving Sharpe unchanged.

- **Limitations:** Treats upside and downside volatility symmetrically, assumes std captures all relevant risk, ignores autocorrelation and fat tails — and an observed Sharpe is a noisy estimate of the true one (the Monte Carlo section quantifies exactly this). For a strategy with rare large losses but otherwise smooth gains, Sharpe can look deceptively attractive.

## Sortino

- **Difference from Sharpe:** The denominator is downside deviation — √(mean over ALL N days of min(r − rf_daily, 0)²) with target = rf_daily — so only below-target returns are penalized. Upside volatility (big positive days) does not hurt the ratio, making Sortino more favorable for right-skewed return distributions.

- **Why full-sample averaging:** Dividing by only the count of down days (a common mistake) shrinks the denominator and inflates the ratio, since a strategy with one terrible day looks identical to one with many moderate down days. My `downside_deviation` function divides by N (all days), not by the count of days where returns fell short.

- **When it tells a different story:** For right-skewed strategies (big up days, small down days) Sortino ≫ Sharpe, because the upside volatility that punishes Sharpe's denominator doesn't enter Sortino's. For a profitable strategy (positive mean excess return) with rare large losses, the opposite holds: the rare disasters dominate the full-sample average in the Sortino denominator even though most days look calm, pushing Sortino below Sharpe.

## Drawdown

- **Max drawdown** is the worst peak-to-trough loss: min of V_t/peak_t − 1, with the running peak including the starting $1 (clipped at 1.0 in `drawdown_series`). A first-day loss already counts as a drawdown under this convention — there is no grace period before tracking begins.

- **Why volatility isn't enough:** Volatility measures typical dispersion around the mean; drawdown measures realized sequences of consecutive losses — the same-volatility portfolios can differ wildly in worst loss depending on whether bad days cluster together. A strategy can have modest volatility but a catastrophic drawdown if losses are autocorrelated.

## Correlation

- Normalized covariance, Cov(p,b)/(σ_p · σ_b), bounded in [−1, 1] by the Cauchy–Schwarz inequality. My implementation computes the sample covariance by hand (`_covariance`) and divides by the product of ddof=1 standard deviations; scipy's `pearsonr` is used in the test suite as an independent cross-check.

- **Zero correlation does not imply independence** — it rules out only linear association. The classic example: y = x² on symmetric x has zero correlation and complete functional dependence. Correlation is a pairwise linear summary, not a general measure of dependence.

## CAPM

- **Beta** = Cov(excess_p, excess_b) / Var(excess_b): the OLS regression slope; sensitivity to benchmark moves. I compute it on excess returns (both series minus daily rf) using the closed-form formula rather than calling a regression library.

- **Beta vs correlation:** beta = ρ · (σ_p / σ_b) — correlation is direction only, beta includes scale. A portfolio with the same direction of movement as the benchmark (ρ = 0.9) but twice the volatility has beta ≈ 1.8, meaning it amplifies benchmark moves rather than merely tracking them.

- **Alpha** = intercept = mean(excess_p) − β · mean(excess_b), annualized ×252 for display: the average daily return not explained by benchmark exposure, scaled to an annual figure. It is an estimate with sampling error — not proof of skill; a short history will produce noisy alpha estimates even for a pure index strategy.

- **R²** = ρ²: the share of portfolio variance the benchmark explains. My `r_squared` function calls `correlation` on raw returns — not excess — which is correct because subtracting a constant rf_daily shifts both series by the same amount and leaves the correlation (and thus R²) unchanged; only alpha is affected by the rf subtraction.

- **Why subtracting rf leaves beta/ρ/R² unchanged:** Subtracting a constant rf_daily from both series shifts their means but not their deviations from those means, so covariances and variances are identical on raw vs. excess returns. Alpha moves because it equals mean(excess_p) − β·mean(excess_b), and both means shift.

## Monte Carlo

- **What it is:** Repeatedly draw outcomes from an assumed probability model to see the distribution of a statistic. Rather than asking "what is the Sharpe?", I ask "across many histories generated by the same process, what is the distribution of measured Sharpes?" — which exposes how much a single history's Sharpe can differ from the true underlying value.

- **Why I used it:** To show that the Sharpe I measured is one draw from a sampling distribution, not the true Sharpe. The dashboard simulates 10,000 alternate histories of the same length as the uploaded data and plots the resulting Sharpe distribution alongside the observed value, making the sampling uncertainty concrete and visual.

- **Where the simulated returns come from:** Normal(μ̂, σ̂²) with parameters fitted to the historical sample (the observed mean and standard deviation). Each simulated history has the same length N as the real one and the same Sharpe formula is applied — so the only source of spread in the histogram is finite-sample randomness, not a different process.

- **Why simulations differ:** Each history is a different finite random sample, so its sample mean and sample std — and hence its measured Sharpe — differ from both the true parameters and from each other. The annualized Sharpe estimate has a standard error of roughly √(252/N); for the bundled 752-day sample that is ≈ 0.58, which is exactly the spread the dashboard's Monte Carlo histogram shows. The uncertainty shrinks like 1/√N with longer histories.

- **Assumptions and why they're imperfect:** Normality (real returns have fat tails and skew), independence (real returns cluster in volatility — ARCH effects), and fixed plug-in parameters (μ̂, σ̂ are themselves uncertain) — all three make the simulated interval an understatement of true uncertainty. The dashboard's EXPLAIN text calls this a demonstration of sampling error, not a forecast.

- **Sample size:** The standard error of the Sharpe estimate shrinks like 1/√N, so longer histories pin it down better. Doubling the history length cuts the sampling uncertainty by ~29%; this is why a 3-year backtest supports a much stronger inference than a 3-month one.

## Dashboard

- **Why Streamlit:** A pure-Python way to publish the analysis as an interactive web app; no frontend stack to maintain. The entire UI is Python, which keeps the codebase uniform and lets the finance logic be unit-tested without any web framework involved.

- **Data flow:** upload → `data_loader` validates → pure functions in `src/` compute → `app.py` arranges → `plots.py` renders. Each stage has a single responsibility: validation, computation, layout, and rendering are fully separated.

- **Why logic is separate from UI:** The finance code is testable (54 tests, including scipy cross-checks) and reusable without Streamlit; `app.py` contains zero math. This separation means I can add a new metric by writing a function in `src/`, writing a test, and wiring a single line into the UI — no risk of mixing concerns.
