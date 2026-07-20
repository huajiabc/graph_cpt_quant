# v10.0 Exact Taker-Flow State Alpha - Pre-Registration

## Scope and permission

This is an offline conditional-alpha test. It cannot change P2, shadow,
paper-live, canary-live, or real-live permissions.

The local Bybit public-trade archive was downloaded for days containing the
existing v0.1 `short_squeeze` or `momentum_ignition` long signals. Therefore it
may test only whether exact pre-signal taker flow improves those independently
defined events. It may not be used to claim an unconditional intraday alpha or
to evaluate minutes before the event that caused a day to enter the archive.

## Frozen event population and chronology

- Event source: deduplicated `(exchange, symbol, path_name, signal_time)` rows
  from `reports/v0_1/entry_policy_1m_trades.csv`.
- Raw source: `data/raw/bybit/public_trading_parquet`; Bybit's reported `side`
  field is treated as taker side.
- Initial population before coverage QA: 596 events.
- Strict as-of feature windows end before `signal_time`; the signal minute and
  all later trades are forbidden as features.
- Entry: first trade/minute open at or after `signal_time`.
- Primary exit: first trade/minute open at or after 60 minutes.
- Secondary diagnostic exits: 15 and 240 minutes. They cannot rescue a failed
  60-minute result.
- A 60-minute cooldown is applied per symbol, preserving the earliest event.
- Development: 2026-04-01 through 2026-04-30.
- Validation: 2026-05-01 through 2026-05-20.
- Holdout: 2026-05-21 through 2026-06-02.

No threshold may change after the state counts or returns are read.

## Frozen exact-flow features

For the strict pre-signal 5-minute and 15-minute windows:

- taker-buy turnover;
- taker-sell turnover;
- signed imbalance `(buy - sell) / (buy + sell)`;
- first-to-last trade return;
- total turnover.

`turnover_acceleration` is five-minute turnover divided by one third of the
fifteen-minute turnover. Minimum coverage is four populated minutes in the
five-minute window and twelve in the fifteen-minute window.

## Frozen states

- `OF1_CONFIRM_LONG` (primary): 5m imbalance at least +0.10, 15m imbalance
  non-negative, 5m price return positive, and turnover acceleration at least
  1.0.
- `OF2_SELL_ABSORPTION_LONG`: 5m imbalance at most -0.10 while 5m price return
  is non-negative. Aggressive selling failed to move price down.
- `OF3_AVOID_BUY_EXHAUSTION`: take every covered event except those with 5m
  imbalance at least +0.10 while 5m price return is non-positive.
- `ALL_COVERED_EVENTS`: frozen event baseline, not a candidate.

OF2 and OF3 are a preregistered family of secondary mechanisms. They must clear
a max-statistic random control across all three candidates; isolated nominal
significance is insufficient.

## Returns and costs

- Gross return is entry-to-horizon price return.
- Net views subtract 10, 20, and 30 bps round-trip cost; net20 is focal.
- Report trade count, symbols, active days, mean/median, win rate, additive
  return, day-block bootstrap confidence intervals, worst day, and maximum
  positive-day contribution.
- Report combined and path-specific (`short_squeeze`, `momentum_ignition`)
  results for development, validation, and holdout.

## Controls

- 500 same-symbol/same-day random 15-minute timestamps with identical strict
  pre-window and forward coverage. Each iteration re-runs all state rules.
- A +60-minute shifted-event timing placebo.
- `ALL_COVERED_EVENTS` to separate flow-state lift from the existing signal.
- Candidate-family max random percentile to control the three frozen states.

## Decision gates

The primary candidate, or either secondary candidate after familywise control,
is at most `research_candidate_only` when all are true:

1. at least 80 full-sample trades, 25 validation trades, and 25 holdout trades;
2. at least six symbols and ten active days;
3. validation and holdout mean net20 are both positive;
4. full-sample mean net30 is positive;
5. day-block bootstrap 95% lower bound for mean net20 is above zero;
6. real mean net20 exceeds the familywise random 90th percentile;
7. real timing exceeds the +60-minute shifted placebo;
8. neither path has negative full-sample mean net20;
9. no positive day supplies more than 35% of total positive additive net20.

Passing remains research-only because coverage spans roughly two months and is
conditional on archived signal days. Failure of every state closes the exact
single-venue taker-flow overlay under this specification; the synchronized
Binance-Bybit tape remains a separate forward-data hypothesis.
