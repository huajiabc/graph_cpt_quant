# v20.4-v20.8 AggTrade Flow-Exhaustion Round Summary

## Outcome

The exact aggTrade sequence inside an extreme community trade-overshoot bar is
informative, but the tested forms do not produce robust deployable alpha under
the frozen 20 bp round-trip cost and timing controls.

No candidate is approved for live, PaperLive, leverage, or natural-forward
observation from this round.

## Data and feature audit

- Collected all 936 frozen receiver windows from 216 source events and 45
  symbols.
- Daily archives were streamed in chunks, target windows were retained, and
  temporary ZIP files were deleted after verified extraction.
- All windows contain at least 204 aggregate trades. First and last trades are
  within eight seconds of their 15-minute boundaries.
- AggTrade first/last price return versus official 15-minute kline open/close
  return: correlation `0.99999823`; p99 absolute difference `1.7204 bp`.
- v20.4 passed 16/16 feature/data checks and froze two structural sign-rule
  candidates before their future-return reveal.

## Preregistered v20.5 reveal

### RFX1 event-wide late-flow reversal fade

- 53 events; gross `+8.9569 bp`; 20 bp net `-11.0431 bp`.
- Development/validation/holdout gross:
  `+10.0783 / +7.4933 / +7.4961 bp`.
- Period-matched random-event percentile: `0.958`.
- Alt contribution `+18.2166 bp`, BTC hedge contribution `-9.2597 bp`.
- One-bar delayed gross `-6.6420 bp`.
- Interpretation: the flow filter selects stronger immediate reversals than
  random extreme events, but the tradable magnitude is below cost and vanishes
  with delayed entry.

### RFX2 exhausted-versus-persistent within-event spread

- 52 events; gross `-1.8519 bp`; 20 bp net `-21.8519 bp`.
- Random-control percentile `0.246`.
- Rejected mechanically and economically.

The v20.6 audit passed 20/20 checks and independently reproduced both
rejections.

## Frozen v20.7 no-hedge diagnostic

RFX3 removed the BTC hedge from the exact RFX1 event set and equal-weighted the
receiver fade bucket. This was explicitly post-hoc because the v20.5
contribution decomposition was already known.

- Full-sample gross `+50.8856 bp`; 20 bp net `+30.8856 bp`.
- BTC-only matched-event gross `+14.1159 bp`.
- Paired receiver-minus-BTC mean `+36.7697 bp`; bootstrap lower 95% bound
  `+7.1303 bp`.
- Holdout gross only `+0.2217 bp`; 20 bp net `-19.7783 bp`.
- One-bar delayed gross `-19.8823 bp`; 20 bp net `-39.8823 bp`.
- Period-matched random-event percentile only `0.762`.
- Receiver primary-net bootstrap lower 95% bound `-4.6069 bp`.

The apparent no-hedge profit is therefore an entry-close/timing-sensitive
short-horizon reversal, not a robust flow-exhaustion alpha. v20.8 passed 20/20
checks and independently reproduced the diagnostic rejection.

## Research implication

The result narrows the search space:

1. Final-third signed flow reversal has event-classification information, but
   not enough standalone economic magnitude after cost.
2. Removing market hedging creates a large in-sample number, but its holdout,
   delay, random-event, and bootstrap behavior fail. It should not be rescued
   with partial-hedge or threshold grids.
3. The next price/volatility branch should focus on propagation topology or
   externally timed shocks, not another transform of same-bar return and taker
   imbalance.

No live, PaperLive, application, leverage, remote, or order state was read or
changed.
