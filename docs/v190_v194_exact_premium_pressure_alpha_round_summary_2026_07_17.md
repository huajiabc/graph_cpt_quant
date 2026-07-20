# v19.0-v19.4 Exact Premium-Pressure Alpha Round Summary

## Scope

This round acquired and audited exact Binance USD-M 15-minute premium-index
OHLC history for the 46-symbol research universe, then tested two frozen alpha
families: OI-unwind reversal confirmed by premium innovation, and price-shock
continuation after opposing premium pressure. No live, PaperLive, leverage,
application, remote, or order scope changed.

## Data result

- 1,667,808 exact rows across 46 symbols were retained with no forward fill.
- 1,182 downloaded Binance archives passed their published SHA256 checksums.
- Research-timestamp exact coverage is 99.996919%.
- The common official 2026-06-29 gap and TON's partial stop are explicit rather
  than imputed.
- v19.0 data audit: 15/15 checks passed.

## v19.2 premium-innovation-confirmed OI unwind

The frozen BTC reversal has 103 events and +7.3275 bp mean gross return, but
-2.6725 bp after the 10 bp primary cost. Development is +2.7281 bp net, while
validation and holdout are -17.7823 and -11.3182 bp. The bootstrap 95% interval
is [-9.5970, +4.1919] bp and the random-family percentile is 79.0%.

The 81-event receiver reversal bucket is already negative gross at -1.8931 bp
and -31.8931 bp after primary cost. Both candidates are rejected. The independent
audit passed 35/35 checks.

Interpretation: exact premium innovation sharpens the old unwind-reversal gross
effect slightly, but does not make it stable, independent, or tradable.

## v19.4 opposing-premium absorption continuation

The frozen BTC continuation has 178 events and -4.5872 bp gross, rejecting the
continuation mechanism directly. Its reversed direction is only +4.5872 bp
gross before a 10 bp cost.

The 69-event receiver bucket has +7.6302 bp gross continuation, with positive
gross means in development (+17.7435 bp), validation (+1.1559 bp), and holdout
(+6.0942 bp). It remains -22.3698 bp after the 30 bp primary book cost and
decays to +3.4561 bp gross at 60 minutes. Only five adjacent events occur within
30 minutes, so stateful overlap cannot materially close the cost gap. Both
candidates are rejected. The independent audit passed 31/31 checks.

Interpretation: premium OHLC pressure shapes contain a real-looking direction
of cross-asset volatility transfer, but at the current event horizon it is a
sub-cost primitive, not an alpha candidate.

## Research boundary and next branch

Do not tune v19.2/v19.4 thresholds or costs to rescue them. The useful retained
facts are:

1. BTC unwind reversal remains a roughly 5-7 bp gross timing primitive.
2. Opposing premium pressure identifies about 8 bp of short-horizon receiver
   continuation, stable in sign but too small for a fresh round trip.
3. Sparse event overlap rules out a portfolio-netting rescue.

The next materially different branch should use exact premium as a slower
cross-sectional relative-value state rather than another sparse event trigger:
residualize each symbol's premium level against BTC and its graph/bucket peers,
form long-discount/short-rich beta-neutral spreads at a frozen lower cadence,
and charge costs on realized turnover plus funding. This tests whether premium
dislocation can accumulate enough convergence/carry over hours to clear costs,
without reusing the failed 15-60 minute event formulation.
