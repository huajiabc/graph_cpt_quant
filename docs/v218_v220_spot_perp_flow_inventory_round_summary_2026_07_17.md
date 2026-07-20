# v21.8-v22.0 Spot-Perpetual Flow-Inventory Round Summary

Verdict: `weak_orthogonal_flow_rank_rejected_standalone_overlay_motivation_only`.

## Question and design

This round tested a previously unexamined information source: causally normalized
Binance spot taker imbalance minus Binance USD-M perpetual taker imbalance.  Spot
and perpetual files intersected the forward monthly graph universe on 61 symbols
from September 2025 through June 2026.

- SFI1 ranked the cross-venue flow gap globally and traded extreme top/bottom
  buckets.
- SFI2 selected the high/low flow-gap pair inside each eligible graph community.
- Decisions occurred only at 00/12 UTC.  Execution waited the next complete hourly
  bar, held 12 hours, used dollar- and BTC-beta-neutral unit-gross perpetual
  portfolios, and charged 20/40 bp round-trip costs.

The no-future v21.8 feature audit passed 17/17 checks.  SFI1 had 362 feature
events and SFI2 had 396, with 10 active months and adequate coverage in
development, validation, and holdout.

## Reveal

| Candidate | Realized events | Gross bp | Net at 20 bp | Random percentile | Bootstrap lower 95%, net bp |
|---|---:|---:|---:|---:|---:|
| SFI1 global | 357 | +4.3989 | -15.6011 | 0.8820 | -21.7662 |
| SFI2 community | 393 | +4.9984 | -15.0016 | 0.9560 | -21.1940 |

SFI2 gross results by development/validation/holdout were +0.3520, +6.7607,
and +9.3036 bp.  The increasing recent response and 0.956 random percentile show
that community-relative spot leadership contains weak orthogonal rank information.
However, it is below the 20 bp book hurdle in every period; even at an assumed
5 bp round trip its full-sample net is approximately zero.  Four-hour and 24-hour
horizons, a one-hour delayed entry, and the +24-hour placebo do not create an
economically viable standalone strategy.

The independent v22.0 audit passed 26/26 checks, reproducing source hash, causal
hourly beta, weights, exact PnL, costs, all 1,000 random paths, block bootstrap,
concentration, and both rejections.

## Research implication

This feature should not be traded or levered independently.  Its only justified
follow-up is a preregistered, no-extra-turnover overlay on an already scheduled
low-frequency strategy such as FSS3/TG1.  That test must ask whether the flow rank
improves existing selection or risk control without creating an additional trade
cycle; threshold-mining the same standalone events is not justified.

No live, PaperLive, application, leverage, remote, or order state was read or
changed.
