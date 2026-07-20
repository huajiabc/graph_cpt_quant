# v9.5 Token DEX Attention -> CEX Underreaction Preregistration

Status: frozen historical research specification. This document does not
change the P2 primary paper ledger, create a paper strategy, or grant live or
canary permission.

## Research question

Can a same-token DEX volume-attention shock forecast positive CEX perpetual
returns when the CEX has not yet expanded and then prints a small positive
confirmation bar?

This is intended to be structurally different from CIC/P2. The event source is
token/pool DEX activity. CEX market breadth, CIC membership, and the P2 market
gate are not required by the candidate.

## Frozen data and timing contract

- DEX events: `token_pool_attention_events.csv`, confidence A/B, hourly sources
  `dexpaprika_pool_ohlcv_1h` and `geckoterminal_pool_ohlcv` only.
- CEX target: Bybit USDT perpetual features only.
- Event construction remains the existing strictly trailing 120-hour volume
  z-score/percentile rule.
- An event becomes visible at `event_time + 65 minutes`.
- Entry context is the first complete 15-minute CEX feature row strictly after
  event visibility, no later than 30 minutes afterward.
- The entry price is that feature row's close. The main outcome is the existing
  `future_ret_12h` label from that close.
- Same-token events are de-overlapped with a fixed 24-hour cooldown before any
  outcome is read.
- The tradable universe is a month-start frozen Top50 by total CEX turnover in
  the prior 30 calendar days. A month needs at least seven prior days of data.
- Mapping and source are logged for attribution; source changes are never
  silently pooled in stability checks.

## Frozen candidates

All thresholds below are fixed before reading candidate outcomes.

1. `TAD0_ALL_ATTENTION`: all eligible de-overlapped token attention events.
2. `TAD1_UNDERREACTION_RECLAIM` (primary hypothesis):
   - `abs(ret_4h) <= 1%` at the CEX context row;
   - `volume_z_1h < 1.0`, so a CEX volume impulse has not already occurred;
   - `ret_15m > 0`, used only as a direction-confirmation/reclaim bar.
3. `TAD2_CEX_CONFIRMATION` (attribution comparator):
   - `ret_1h > 0`;
   - `volume_z_1h >= 1.0`;
   - `ret_4h <= 4%`.
4. `NEG_UNDERREACTION_DOWN_BAR` (direction negative control): the TAD1
   underreaction state with `ret_15m <= 0`.

No threshold search, score fitting, symbol exclusion, source exclusion, or
exit tuning is permitted after outcomes are read. A later change requires a
new version and is evaluated as a new hypothesis.

## Cost and split contract

- Gross: `future_ret_12h`.
- `net10/net20/net30/net50`: gross minus two times the stated per-side basis
  point assumption.
- Search: entry before 2026-02-01 UTC.
- Validation: 2026-02-01 through 2026-04-30 UTC.
- Holdout: entry on or after 2026-05-01 UTC.
- The vendor/source transition between older Dexpaprika history and newer
  GeckoTerminal history is treated as a robustness test, not normalized away.

## Required controls and audit outputs

- Deterministic seven-day shifted-event placebo.
- Same-token, same-month random-time distribution.
- Same-chain random-token, same-time distribution where a mapped alternative
  exists.
- Search/validation/holdout results.
- Source, month, and symbol attribution.
- Leave-one-month and 35% month-capped net20.
- Entry-day block-bootstrap 95% confidence interval on mean net20.
- P2 temporal overlap: a TAD entry is marked overlapping when the same token has
  a P2 entry within plus/minus 60 minutes.

## Frozen decision rule

`TAD1_UNDERREACTION_RECLAIM` may become a forward paper counterfactual only if:

1. full, validation, and holdout net20 are all positive;
2. validation and holdout each contain at least 20 trades and two active months;
3. full-sample month-cap35 net20 is positive and maximum month contribution is
   no more than 35%;
4. full-sample bootstrap 95% lower bound on net20 is above zero;
5. real net20 exceeds the 90th percentile of same-token random time and the
   median same-chain random-token control;
6. the seven-day shifted placebo net20 is lower than real net20;
7. after removing plus/minus-60-minute P2 overlaps, at least 50% and at least
   20 TAD1 trades remain, and their net20 remains positive; and
8. net30 remains positive.

Failure of any rule keeps the candidate offline. Failure caused only by sample
size is `insufficient_sample`, not a positive or negative alpha verdict.
