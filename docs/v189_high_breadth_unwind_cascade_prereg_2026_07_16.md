# v18.9 High-Breadth Unwind Cascade Preregistration

Status: frozen after feature-only coverage audit and before inspecting future
candidate returns.

## Feature-only decision

The prior-30-day q30 low-breadth regime produced only 15 q85 unwind events
(11/2/2 across development/validation/holdout), so the proposed isolated-shock
reversal candidate is cancelled before any outcome inspection.

The q70 high-breadth regime produced 275 q85 unwind events (139/51/85). Its
median event has 28 of 45 altcoins moving more than one prior-volatility standard
deviation in the BTC source direction. This is the only primary candidate.

## Frozen breadth and source

- Reuse the exact v18.5 BTC unwind event with q85 absolute-return threshold.
- Estimate each alt's return volatility per calendar month from only the prior
  30 days, requiring at least 2,000 paired BTC/alt bars.
- At the source close, standardize each alt return by frozen volatility and align
  it to the BTC source sign.
- Volatility breadth is the fraction of valid alts whose aligned standardized
  return exceeds 1.0.
- High breadth requires the current value at or above its shifted prior-30-day
  q70, with at least 20 days of breadth history.
- Exact completed 15-minute timestamps only; no forward fill.
- Development is before 2026-01-01 UTC, validation is January-February 2026,
  and holdout is March 2026 onward.

## Frozen candidate

`VBR1_HIGH_BREADTH_UNWIND_CASCADE_CONTINUATION` trades BTC in the source
price/flow direction. The primary hold is one bar / 15 minutes because the
mechanism is continued liquidation or short-cover propagation, not slow mean
reversion. Primary/stress round-trip costs are 10/15 bp.

## Frozen controls and gates

- Exact reversed direction and one-bar delayed entry.
- The non-high-breadth q85 unwind complement is the regime control.
- q90 source events and 30/60-minute holds are diagnostics.
- 500 deterministic month-matched random q85 unwind subsets preserving the
  selected event count per calendar month.
- Day-block bootstrap with 2,000 iterations.
- At least 150 full-sample, 20 validation, and 40 holdout events.
- Positive primary mean in development, validation, and holdout.
- Positive full-sample stress mean and bootstrap 95% lower bound.
- At or above the random-control 95th percentile.
- Beat reversed direction, one-bar delay, and the non-high-breadth complement.
- q90, 30-minute, and 60-minute diagnostics remain positive after primary cost.
- No single profitable month supplies more than 35% of positive monthly PnL.

Passing creates an offline research candidate only. No PaperLive, live,
application, leverage, remote, or real-order scope may change.
