# v9.5.1 Attention + CEX Confirmation Attribution Preregistration

Status: post-v9.5 attribution follow-up. The positive TAD2 summary is already
known. This document freezes attribution tests; it is not a blind discovery
claim and it grants no paper/live permission.

## Frozen candidate

`TAD2_CEX_CONFIRMATION` is unchanged from v9.5:

- same-token hourly DEX volume-attention event, visible after 65 minutes;
- month-start frozen prior-30-day turnover Top50;
- same-token 24-hour event cooldown;
- first complete Bybit 15-minute feature row after visibility;
- `ret_1h > 0`, `volume_z_1h >= 1.0`, and `ret_4h <= 4%`;
- 12-hour outcome and the existing two-sided cost convention.

No threshold, symbol, source, entry, or exit change is allowed in this
follow-up.

## Attribution question

Does the DEX event add information beyond the already-positive CEX impulse
state, or is TAD2 only another CEX momentum sample?

The frozen controls are:

1. same-token, same-month random CEX rows satisfying the exact TAD2 CEX state,
   excluding plus/minus 24 hours around the real event;
2. same-chain random-token, same-time events, followed by the exact TAD2 state;
3. the same event shifted forward seven days;
4. removal of same-token P2 entries within plus/minus 60 minutes.

## Decision rule

TAD2 can only remain an alpha research candidate if real net20 beats the 90th
percentile of same-token random time, the median same-chain random-token
control, and the shifted-event placebo. Net20 after removing P2 overlaps must
remain positive with at least 50% and at least 20 trades.

Even if attribution passes, v9.5's original promotion rules still apply. In
particular, the current validation/holdout sample minimum, month contribution,
day-block bootstrap, and net30 gates are not waived.
