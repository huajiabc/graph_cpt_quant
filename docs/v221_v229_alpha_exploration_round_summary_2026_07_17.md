# v22.1--v22.9 Alpha Exploration Round Summary

Date: 2026-07-17

## Outcome

This round completed three preregistered, causal alpha branches. None is
eligible for strategy promotion or deployment. One new information primitive
survived: synchronized multi-alt book pressure plus broad depth withdrawal
forecasts a right-tailed increase in future BTC realized variance. It does not
forecast an executable futures direction or cross-sectional spread.

| Branch | Main result | Control evidence | Verdict | Audit |
|---|---:|---:|---|---:|
| SFI rank tilt inside weekly FSS3 | `-5.69 bp/week` active primary increment | random-rank percentile `26.1%`; reversed better | reject | 23/23 |
| 16-alt book vacuum direction -> BTC | `+2.82 bp` gross, `-7.18 bp` net10 per 4h event | random-time percentile `39%`; delayed and no-vacuum better | reject | 16/16 |
| Vacuum-event Top4-minus-Bottom4 alt spread | `-7.63 bp` gross, `-37.63 bp` net30 per 4h event | random-rank percentile `1.7%`; all periods/horizons gross negative | reject | 15/15 |

## What was genuinely learned

1. **The weekly SFI transfer failed at the signal level, not just cost.** The
   overlay's price selection lost `6.09 bp/week`; funding added only `0.80 bp`.
   Development, validation and holdout increments were all negative.
2. **Broad book withdrawal contains volatility information.** Across 159
   feature-audited events, future/prior BTC four-hour realized-variance ratio
   averaged `1.77x`, including `2.34x` in validation and `1.47x` in holdout.
   The median was `0.84x`, so the forecast is a right-tail warning rather than
   uniform expansion.
3. **Contemporaneous book direction is not the missing key.** BTC one-hour gross
   was `-11.80 bp`; four-hour gross recovered only to `+2.82 bp`, below cost and
   unstable out of development. A one-hour-delayed trade was better.
4. **The variance state does not create relative-return diffusion.** Ranking the
   same 16 coins by their causal imbalance produced negative gross spreads at
   1h (`-2.43 bp`), 4h (`-7.63 bp`) and 8h (`-10.33 bp`). It ranked below 98.3%
   of arbitrary within-event rank assignments.

## Research boundary

More thresholds, stronger depth multipliers, a different nearby holding hour,
or a complex classifier on these same outcomes would be post-result search.
The failures are not caused by beta leakage, gross normalization, price
alignment, costs, random-control implementation or summary arithmetic; all
three branches were independently audited.

The volatility primitive can only become alpha through a payoff that actually
owns convexity or variance. The current historical depth window does not have
overlapping executable option straddle quotes, spreads and delta-hedging costs.
The defensible continuation is therefore new synchronized data: forward option
surface/straddle execution recording or liquidation/forced-flow tape. Until
then, the book-vacuum state may be retained as a forward risk-warning feature,
not a directional entry or levered sizing signal.

No live, PaperLive, leverage, remote, application, or order state changed.
