# v11.6 Community Volatility Path-Efficiency Findings

Date: 2026-07-15

Verdict: `reject_community_volatility_path`.

The exact frozen 8x9 monthly communities produced 412 high-volatility, high-path-efficiency
continuation portfolios across 217 days and eleven months. The implementation rebuilt 15-minute
returns directly from the original 73-symbol Bybit kline set; an earlier sparse-panel smoke run
was discarded before formal controls because it did not satisfy the preregistered universe.

| Scope | Observations | Gross 4h | Net 20 bp | Residual gross 4h |
|---|---:|---:|---:|---:|
| All | 412 | +4.15 bp | -15.85 bp | +10.20 bp |
| Development | 173 | +4.92 bp | -15.08 bp | +3.65 bp |
| Validation | 126 | +16.58 bp | -3.42 bp | +24.97 bp |
| Holdout label | 113 | -10.90 bp | -30.90 bp | +3.77 bp |

The event timing was much better than the one-day-shifted control (-39.40 bp net 20), but the
real communities were worse than random exact-size partitions. Fifty random partitions averaged
-7.81 bp net 20; the real graph ranked at only the 8th percentile. The bootstrap 95% interval was
[-35.33, +5.90] bp, three of five chronological slices were negative, and the largest positive
month contributed 52.25% of positive PnL.

Post-hoc quintile diagnostics did not support another threshold follow-up. Higher volatility
severity had some positive full-sample shape, but the top quintile was positive in development
and validation and negative in the holdout label. Path efficiency itself was not monotonic.

Interpretation: coordinated one-sided community paths contain a small amount of timing
information, but not enough to cover costs, and the balanced price graph does not identify the
profitable source. This feature may eventually be useful for holding-period or risk management of
an independently valid entry; it is not an independent alpha.

No PaperLive, leverage, or real-order permission changed.
