# v14 High-/Secondary-Frequency Graph-Alpha Round Summary

Date: 2026-07-15

## Bottom line

The high-/secondary-frequency price, flow, volatility, and graph branches did not produce a new
tradable mechanism. Reusing that negative evidence redirected the search to settled funding
cashflow and crowding state. The resulting v14.9 `FSS3_CURRENT_SIGN_070_TURNOVER_CAP` passed all
frozen gates and an independent raw-data audit, so it is frozen as a forward-shadow candidate.
It does not earn PaperLive, leverage, or live-order permission in this round.

## Frozen studies

| Version | Mechanism | Main sample/result | Structural controls | Verdict |
|---|---|---|---|---|
| v14.1 | 15m Binance taker-flow leader -> 1h follower return graph | Positive: 660 events, residual net40 -40.64bp. Negative: 715, -39.29bp. | Random-family percentiles 22%/50%; reversed and delayed do not rescue it. | Reject both directions. |
| v14.2 | Broad community volatility release -> 4h signed follower bucket | Continuation: 201, gross residual +11.49bp, net40 -28.51bp, no validation events. Reversal: 482, gross -7.56bp. | Random percentiles 60%/0%; unstable monthly edge sign. | Reject. |
| v14.3 | Quiet receiver vs shocked source convergence spread | 4h/8h/12h gross residual +0.76/+13.31/+22.05bp; 12h net40 -17.95bp on 90 events. | 12h random percentile 94%, but validation and holdout net40 remain negative. | Reject family. |
| v14.4 | Walk-forward nonlinear model on broader bearish graph pairs | 2,329 pair rows; 34 trades; gross residual +30.65bp, net40 -9.35bp. Development/validation positive, holdout -36.72bp net40. | Label-null percentile 98%; beats static/reversed/delayed, but bootstrap lower -44.94bp and threshold curve fails throughout holdout. | Reject model; learned relationship is nonstationary. |
| v14.5 | Exact bearish convergence extended to 18h/24h | 18h: 42, net40 -31.05bp. 24h: 40, net40 -29.13bp. Validation net40 -97.87/-113.46bp. | Random percentiles 52%/58%; 24h fails delayed control. | Reject; the 12h bump does not continue. |
| v14.6 | Exact temporal extension of frozen v11.2 high-vol topology break | Canonical parity 213/213, max return drift 0. New 4h: 7 events over June/July, gross -13.50bp, net20 -33.50bp. | Full 4h remains random percentile 98%, but new bootstrap net20 is [-96.70,+34.48]bp. | Insufficient confirmation; downgrade v11.2 evidence. |
| v14.7 | Long all negative-funding names, short all positive-funding names, exact BTC-beta hedge | +90.99bp/week net20; validation +1.84bp; bootstrap lower +24.58bp; family-null 100%. | Passed every frozen economic/robustness gate except mean turnover 0.7595 > 0.75. | Near-candidate; do not promote unchanged. |
| v14.8 | Require two consecutive opposite funding signs before switching side | Turnover 0.3196, but validation -55.19bp and bootstrap lower -36.34bp. | Null percentile 92.6%; return correlation with v14.7 only 0.3403. | Reject; stale-side retention destroys the signal. |
| v14.9 | Exact v14.7 target with full-L1 weekly transition cap 0.70 | +94.17bp/week net20; stress +80.80bp; development/validation/holdout +164.45/+5.78/+65.80bp. | Turnover 0.6685; bootstrap lower +29.34bp; null 99.7%; month concentration 24.17%; exact beta/gross. | Freeze as forward-shadow candidate. |
| v15.0 | Independent raw-data and statistical audit of v14.9 | Max reconstruction drift 8.0e-15; alternate bootstrap lower +29.41bp; 5,000-path null 99.64%. | Raw funding and price diffs exactly zero; all frozen gates independently re-passed. | Audit pass. |

## Mechanism conclusions

1. Continuous taker-flow shocks do not produce a useful symbol-level directed return graph. The
   edge-learning layer performs no better than randomized leaders.
2. Community aggregation reveals genuine volatility transmission, but unsigned volatility is much
   easier to predict than a tradable direction. Directional follower buckets deliver only
   10-30bp gross on average.
3. Bearish source shocks briefly create a 12-hour source/receiver convergence spread. The
   result-informed negative subset had 39 events and +52.32bp gross residual, but its bootstrap
   lower bound was negative and holdout net40 was slightly negative. At 18/24 hours it decayed.
4. A nonlinear walk-forward model separated historical winners from label-null models, yet failed
   in the latest holdout at every examined prediction hurdle. More model complexity does not repair
   the state drift.
5. Exact new data weakened the prior strongest v11.2 result. Its full-history graph specificity is
   real, but the payoff remains a rare convex interaction rather than a stable average-return
   process.
6. Funding sign contains a broader cashflow/crowding effect than funding severity or price-based
   graph topology. The usable construction needs both sides: negative-funding longs collect carry
   and rebound exposure, while positive-funding shorts improve the price spread. A causal execution
   cap fixes the small turnover breach without changing the contemporaneous signal.

## Research governance

- All v14.1-v14.9 hypotheses were written before their respective returns were inspected; adaptive
  follow-ups v14.3-v14.5 are explicitly labeled and cannot independently establish alpha.
- Real graphs were compared with random, reversed, delayed, or label-permuted controls as
  appropriate.
- Costs, cooldowns, time splits, daily block bootstrap, and concentration gates were retained.
- v11.2 extension parity is exact, so its weaker new result is not a pipeline drift artifact.
- v14.9 was independently rebuilt from raw settled funding and closed one-hour price endpoints;
  saved and reconstructed weekly results differ by no more than 8.0e-15.
- PaperLive, leverage, remote-host, and real-order permissions were not changed.

## Next defensible work

Run v14.9 only as a natural forward shadow with persisted prior executed weights, exact data-cutoff
telemetry, and the 0.70 transition cap. Its validation mean is positive but thin, and the historical
worst week/max additive drawdown are -356.59/-712.98bp, so leverage and PaperLive require separate
evidence. Do not mine additional funding thresholds or cap values on this archive. Further price/
flow work should wait for genuinely new events or orthogonal liquidation/order-book data.
