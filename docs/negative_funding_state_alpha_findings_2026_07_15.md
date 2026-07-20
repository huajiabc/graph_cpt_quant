# Negative-Funding State Alpha Findings

Date: 2026-07-15

## Outcome

The research produced the first candidate in this round to pass every frozen retrospective
promotion gate: `NF8_ALL_NEGATIVE_EQUAL_BTC_BETA_NEUTRAL` (v14.0).

Its mechanism is deliberately simple. Every Monday, hold all frozen-universe Bybit perpetuals
whose preceding seven-day settled funding sum is negative, equal weight them, and short BTC by the
basket's causal trailing beta. Gross notional is one. No graph, rank, top-k rule, OI, flow, basis,
price, volatility, or regime filter remains in the final signal.

## Exploration path

| Version | Mechanism | Key result | Verdict |
|---|---|---:|---|
| v13.3 | Seven weekday TG1 carry ladder | validation -18.77 bp; bootstrap lower -0.31 bp | Reject |
| v13.4 | Fixed nine-name negative-funding basket + BTC hedge | +153.48 bp; bootstrap lower +36.66 bp; null 90.2% | Near-candidate; only 39 weeks/10 months |
| v13.5 | Adaptive four-to-nine most-negative names | +147.50 bp; bootstrap lower +52.42 bp | Reject; null 89.4% vs 90% gate |
| v13.6 | All negative names, severity weighted | contracted breadth -125.93 bp; null 38.3% | Reject |
| v13.7 | Bybit/Binance dual-negative confirmation | contracted breadth -48.40 bp; null 78.9% | Reject |
| v13.8 | Negative funding split by OI build/unwind | OI-build +148.03 bp, but 40 weeks, turnover 0.853, family null 90.3% | Reject family gate |
| v13.9 | Funding/OI/taker continuous pressure rank | all periods/states positive; null 44.7% | Reject ranking attribution |
| v14.0 | Equal weight every negative-funding name + BTC hedge | all frozen gates pass | Promote to forward-shadow candidate |

## v14.0 frozen evidence

| Metric | Result |
|---|---:|
| Weeks / months | 49 / 12 |
| Primary / stress net | +102.29 / +93.22 bp per week |
| Development / validation / holdout | +151.21 / +28.53 / +93.91 bp |
| Contracted 4-8 / broad 9+ states | +98.75 / +103.20 bp |
| Four-week block bootstrap 95% lower | +38.37 bp |
| Full-universe random-basket percentile | 99.9% |
| Positive-month concentration | 28.28% |
| Mean exact weekly turnover | 0.4535 |
| Maximum residual estimated BTC beta | 2.78e-16 |

The independent audit reconstructed raw funding windows, price endpoints, holdings, weights,
funding cash flow, BTC hedge, turnover, and costs with maximum numerical drift below `5e-16`.
Using a different seed, 10,000 block-bootstrap draws gave a +38.98 bp lower bound and 5,000
full-universe random paths placed the candidate at the 99.96th percentile. The latest six weeks
averaged +46.83 bp with four positive weeks.

The realized weekly price beta to BTC was -0.0168 and price/BTC correlation was -0.0299. Returns
were positive in both BTC-up and BTC-down weeks. Funding alone, after primary and stress turnover
costs and with price PnL set to zero, remained +30.09 and +21.02 bp per week. The strongest
positive altcoin contributed 19.20% of positive coin contribution, so the result is neither a
disguised BTC short nor a single-altcoin outcome.

## Permission boundary

v14.0 is a **usable forward-shadow candidate**, not a live strategy. It has not been installed in
PaperLive, is not authorized for leverage, and grants no real-order permission. A deployment,
execution implementation, or leverage study requires its own frozen specification and explicit
user request.
