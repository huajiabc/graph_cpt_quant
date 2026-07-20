# v11.2 High-Volatility Topology-Continuation Preregistration

## Status

This is a second result-informed mechanism follow-up. The high-volatility interaction was observed
in v11.1 diagnostics. The entire replay is therefore retrospective and cannot authorize PaperLive,
even if all numerical gates pass.

## Frozen candidate

Reuse the v11.1 balanced communities, base topology-break event, expanding prior-month 80th
percentile severity rule, and continuation spread. Add exactly one market-state condition:

- At each month start, calculate BTC hourly-return rolling 24-hour volatility over the preceding
  30 calendar days.
- Freeze its 75th percentile for the following month.
- Enter a qualifying sparse topology event only when its as-of 24-hour BTC volatility is at or
  above that frozen threshold.

No community-count, severity, breadth, direction, or single-leg branch is added. At most three
simultaneous community sleeves remain allowed by severity.

## Horizons, controls, and costs

- Evaluate the fixed 1, 2, 4, 8, and 12-hour continuation spreads.
- Apply 20, 30, and 50 bp round-trip spread costs.
- Rebuild fifty exact-size random monthly partitions. Each learns its own expanding severity
  threshold and receives the same external BTC-volatility gate.
- Rebuild the one-day shifted signal and apply the same gate.
- Compare each real horizon with the best net-20 result across all five random horizons.
- Run two thousand entry-day bootstrap resamples, five chronological slices, and the same 35%
  positive-PnL month/community concentration limits.

The economic gates remain unchanged, including at least 100 observations overall and at least 25
in validation and holdout labels. They are deliberately not relaxed to accommodate the smaller
high-volatility sample.

Passing can only produce `retrospective_forward_watch_only`. Failure rejects this exact nested
severity-plus-volatility rule, not the underlying topology mechanism.
