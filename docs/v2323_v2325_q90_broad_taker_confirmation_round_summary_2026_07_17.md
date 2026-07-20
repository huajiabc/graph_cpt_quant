# v23.23--v23.25 q90 Broad-Taker Confirmation Round Summary

This round tested whether the post-selected positive-q90 book-pressure event
could be converted from a directionless breakout into directional BTC alpha by
requiring independent broad aggressive buying. No live, PaperLive, leverage,
remote, application, or order state changed.

## Frozen feature

Every one of the 53 positive-q90 events was joined to the exact completed
Binance five-minute metrics timestamp for the same 16 symbols. The single rule
required at least 9/16 taker long/short volume ratios above one.

- Confirmed events: 26 across 10 months.
- Development / validation / holdout: 8 / 7 / 11.
- Median confirmed breadth: 10/16.
- v23.23 feature audit: 14/14 checks passed.

## Directional reveal

| Scope | Events | Gross long | Net at 10 bp | Matched-random percentile |
|---|---:|---:|---:|---:|
| all | 26 | -6.06 bp | -16.06 bp | 41.2 |
| development | 8 | +10.24 bp | +0.24 bp | 67.9 |
| validation | 7 | -22.96 bp | -32.96 bp | 29.3 |
| holdout | 11 | -7.17 bp | -17.17 bp | 42.4 |

The 15-minute delayed result was worse at -22.38 bp net. The 27 unconfirmed
q90 events returned -13.35 bp net, so confirmation did not improve the full
sample and was materially worse in validation. The within-month confirmation
label permutation p-value was 0.6812. The absolute month-bootstrap lower bound
was -36.82 bp, every leave-one-month-out mean was negative, and positive monthly
PnL concentration was 81.5%.

Two holdout events had fewer than the preregistered five exact-hour controls;
this was retained as a failed gate rather than loosening the match rule after
reveal. Eleven of twelve evidence gates failed. v23.25 independently rebuilt
the 3,318-hour control universe, 219 control links, 4,000 random paths, 5,000
bootstrap draws, 5,000 label permutations, and all decisions; 11/11 audit
checks passed.

## Interpretation

Positive book pressure plus aggressive buying is not a BTC continuation signal.
The result rejects the simple explanation that q90 works because broad buying
predicts an upward move. The surviving q90 evidence remains path-dependent:
extreme book pressure identifies a state in which a narrow OCO can let the
market choose direction. Replacing that convex payoff with a directional long
trade destroys the edge.

No additional breadth threshold, flow/position combination, or sign flip is
authorized on this history. The next evidence for q90 must come from genuinely
new forward events, while a new standalone branch requires a new pre-price data
source rather than another transformation of the existing Binance metrics.
