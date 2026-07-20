# v12.7 Broader Cross-Venue Carry Bucket Preregistration

Date frozen: 2026-07-15, before inspecting any v12.7 return.

## Mechanism

v12.6 passed every promotion gate except positive-month concentration. Attribution
showed that three extreme funding-spread names dominated November/December. The
next and only allowed refinement broadens the frozen multi-coin bucket rather
than clipping realized profits or tuning a return threshold.

`BD1_30D_TOP12_HOLD24` reuses the unchanged v12.5 weekly panel, same-coin
Bybit-long/Binance-short return, 30-day cumulative funding-spread score, and
chronology. It starts or fills from the top 12 positive scores, retains a name
while its score remains positive and its current rank is no worse than 24, and
holds twelve equal 1/12 weights. A week with fewer than twelve positive names is
not emitted and the next valid week restarts. No basis, volatility, community,
or realized-return filter is permitted.

## Costs, controls, and gates

- Primary cost: 20 bp one-way times realized name-weight turnover, including
  initial entry and terminal close.
- Stress cost: 40 bp one-way times the same path.
- 2,000 week-block bootstrap resamples of primary net return.
- 1,000 within-week random positive-spread 12-name baskets using the exact
  observed weekly cost path.
- Development/validation/holdout boundaries are unchanged.

Promotion requires at least 40 weeks, ten months, ten validation weeks, and
eight holdout weeks; positive primary net return in all three periods; positive
full-sample stress net and funding contribution; positive bootstrap 95% lower
bound; null percentile at least 90; positive-month concentration no greater than
35%; worst period no worse than -40 bp/week; and mean weekly turnover no greater
than 0.50.

Passing means forward shadow candidacy only. No PaperLive strategy is modified.
