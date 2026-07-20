# v12.1 Top-Trader Divergence and Community Rotation Preregistration

Date frozen: 2026-07-15, before downloading the historical Binance USD-M
metrics panel or inspecting any candidate return.

## Question

Does a divergence between large-trader positioning and the broad account
population lead a subsequent four-hour return in individual multi-coin
buckets or in the already-frozen price communities?

The test is deliberately about transmission and portfolio formation. It is
not another threshold search on the existing Bybit account-ratio signal.

## Data and as-of contract

- Existing Bybit 15-minute feature panel and its `future_ret_4h` labels.
- Existing v11.0 monthly balanced memberships: eight communities of nine
  symbols, frozen from data strictly before each target month.
- Binance USD-M daily `metrics` archives from 2025-07-01 through the last
  common feature day. The archive has five-minute observations of large-trader
  account long/short ratio, large-trader position long/short ratio, all-account
  long/short ratio, taker long/short volume ratio, and open interest.
- At a decision time `t`, the newest admissible metrics observation is at or
  before `t - 5 minutes`. This lag removes ambiguity about interval-close
  publication. No forward fill may cross 15 minutes.
- Signals are evaluated only at 00:00, 04:00, 08:00, 12:00, 16:00, and 20:00
  UTC. Four-hour holdings therefore do not overlap.

The exact monthly 72-symbol membership remains the target universe. Symbols
without a Binance USD-M archive are unavailable rather than retrospectively
replaced. A timestamp requires at least 48 covered symbols. A frozen community
requires at least six of its nine members.

## Causal transforms

For symbol `i` at time `t`:

- `D = log(top-trader position ratio / all-account ratio)`;
- `S = log(top-trader position ratio / top-trader account ratio)`;
- `F = log(taker long/short volume ratio)`.

Each raw feature is converted to a per-symbol trailing 30-day z-score using
only observations before the current observation (`shift(1)`), with at least
20 days of hourly history. Z-scores are clipped to `[-5, 5]`.

The active-flow score is the equal-weight mean of the cross-sectional
percentile ranks of `z(D)` and `z(F)`. Equal weights are frozen; no fitted
coefficient is allowed.

## Candidate families

1. `TD1_POSITION_VS_CROWD`: rank symbols by `z(D)`; hold the top nine against
   the bottom nine.
2. `TD2_POSITION_VS_TOP_ACCOUNTS`: rank symbols by `z(S)`; hold the top nine
   against the bottom nine.
3. `TD3_DIVERGENCE_PLUS_TAKER_FLOW`: rank the active-flow score; hold the top
   nine against the bottom nine.
4. `TD4_FROZEN_COMMUNITY_ROTATION`: take the median `z(D)` inside each eligible
   frozen community; hold all covered members of the highest-scoring community
   against all covered members of the lowest-scoring community.

Portfolio weights are `+0.5` equal-weight in the high bucket and `-0.5`
equal-weight in the low bucket. Consequently gross exposure is one and net
market exposure is zero.

The economic direction is not assumed. For each family independently, choose
continuation or reversal solely from mean development-period gross return. A
tie chooses continuation. That sign is then frozen for validation and holdout.
No family, sign, bucket size, horizon, transform, or timestamp may be changed
after candidate returns are inspected.

## Chronology

- Development and sign selection: 2025-08-01 through 2025-12-31.
- Validation: 2026-01-01 through 2026-03-31.
- Untouched holdout: 2026-04-01 onward.

History from July 2025 is warm-up only.

## Costs and diagnostics

- Primary conservative result: subtract 40 bp per four-hour observation,
  equivalent to a full round trip at 20 bp one-way on unit gross exposure.
- Stress result: subtract 60 bp per observation.
- Also report realized weight turnover at 20 bp one-way. This is descriptive;
  it cannot replace the conservative promotion gate.
- Report raw and trailing-BTC-beta-residual portfolio returns, cross-sectional
  Spearman IC, decision count, month count, coverage, win rate, worst period,
  maximum drawdown, monthly and symbol contribution concentration.
- Day-block bootstrap: 2,000 resamples of daily mean net-40-bp returns.
- Nulls: 200 within-timestamp random bucket permutations for TD1-TD3; 100
  random nine-symbol community partitions for TD4. Random partitions preserve
  monthly membership and bucket sizes.

## Promotion gate

A family may enter forward shadow review only if all conditions hold after its
development-only sign is frozen:

1. at least 500 decisions, ten active months, and median coverage of at least
   60 symbols;
2. mean net 40 bp is positive in development, validation, and holdout;
3. full-sample mean net 60 bp and BTC-residual net 40 bp are positive;
4. the 95% day-block-bootstrap lower bound for net 40 bp is positive;
5. observed net 40 bp exceeds the 90th percentile of its applicable null;
6. no month supplies more than 35% of positive PnL and the worst chronological
   period is not worse than -40 bp per decision.

Failure means rejection as a tradable alpha. Predictive but sub-cost behavior
may be retained only as a risk or execution-state feature. Existing PaperLive
strategies are not modified by this study.
