# v17.3 Deribit Skew-to-Receiver-Bucket Preregistration

Status: `PREREGISTERED_OFFLINE_RESEARCH_ONLY`

No candidate outcome was inspected before this rule set was frozen.

## Mechanism

A sudden richening of BTC downside option volatility is treated as an external
stress source. The tradable receivers are not fixed narratives or hand-picked
coins: each calendar month they are the alt perpetuals with the strongest
causally estimated lagged absolute-return transmission from BTC over the prior
90 days. The primary question is whether the option surface leads a bucket of
high-sensitivity receivers after ordinary BTC price information is accounted
for.

## Inputs

- Deribit v17.2 quality-passing daily trade surface.
- Binance USD-M 1h closed bars for BTCUSDT plus:
  ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT,
  AVAXUSDT, LTCUSDT, BCHUSDT, and ETCUSDT.
- Research range: 2021-03-01 through 2026-06-30 UTC.
- Development: through 2023-12-31.
- Validation: 2024-01-01 through 2024-12-31.
- Holdout: 2025-01-01 onward.

## Causal surface signal

1. Use `downside_risk_reversal = put25_iv - call25_iv`.
2. Compute one-row innovations only within the same expiry and only when adjacent
   feature timestamps are at most two days apart.
3. Standardize the innovation with the prior 120 valid observations using the
   shifted rolling median and `1.4826 * MAD`; require at least 40 observations.
4. Stress event: risk-reversal robust z-score at least +1.0 and ATM-IV innovation
   strictly positive.
5. Relief event: risk-reversal robust z-score at most -1.0 and ATM-IV innovation
   strictly negative.
6. Apply a three-calendar-day global cooldown. The completed UTC day is known at
   `feature_time`; entry is the then-current closed hourly price and exit is 24
   hours later.

## Monthly receiver graph

- Estimate at each month start from the preceding 90 calendar days only.
- Hourly log returns are used.
- For each alt, calculate:
  - forward edge = correlation of `abs(BTC[t-1])` with `abs(alt[t])`;
  - reverse edge = correlation of `abs(alt[t-1])` with `abs(BTC[t])`;
  - direction advantage = forward minus reverse.
- Require at least 1,000 paired hours and positive forward edge.
- Rank by direction advantage, then forward edge; take the top four. Require at
  least three receivers.
- Estimate the receiver bucket's contemporaneous BTC beta on the same trailing
  window and clamp it to [0.5, 2.0] only for the neutral candidate.

## Frozen candidates

- `DSR1_STRESS_RECEIVER_SHORT`: on stress events, equally short the receiver
  bucket for 24h. Primary net cost: 20 bp; stress cost: 40 bp.
- `DSR2_STRESS_RECEIVER_BTC_NEUTRAL`: same short bucket plus a beta-scaled long
  BTC leg. Primary net cost: 30 bp; stress cost: 50 bp.
- `DSR3_RELIEF_RECEIVER_LONG`: on relief events, equally long the receiver bucket
  for 24h. Symmetry diagnostic; it cannot promote by itself in this round.

## Required controls

- Same-size random receiver buckets, 500 iterations; compare the family maximum.
- Signal shifted forward by one day as a timing placebo.
- Robust-z thresholds 0.75 and 1.25 as fixed sensitivity checks.
- Year, development, validation, and holdout decomposition.
- Event-day block bootstrap with 2,000 iterations.
- Report gross, primary-cost, and stress-cost returns; no leverage.

## Candidate gate

Only DSR1 or DSR2 can remain a research candidate, and only if all hold:

- at least 30 full-sample, 5 validation, and 10 holdout events;
- primary-cost mean is positive in full, validation, and holdout;
- stress-cost full-sample mean is positive;
- 95% event-block bootstrap lower bound is positive;
- random-bucket family percentile is at least 90%;
- real timing beats the one-day shifted placebo;
- both threshold sensitivities retain positive primary-cost means;
- no single positive year contributes more than 50% of total positive PnL.

Passing this audit creates an offline research candidate only. It does not grant
PaperLive, application, remote, leverage, or real-order permission.
