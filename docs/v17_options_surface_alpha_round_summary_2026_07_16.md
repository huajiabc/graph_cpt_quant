# v16.7--v17.1 Options-Surface Alpha Round Summary

Date: 2026-07-16

## New auditable data

The official Binance archive was expanded beyond the previously used DVOL index:

- 145 valid `BTCUSDT` option EOH days from 2023-05-18 through 2023-10-23;
- 32,972 retained hour-0 option rows with bid/ask, IV, Greeks, volume and OI;
- 122 days with at least one complete 21--45 DTE call/put pair;
- 13 complete USD-M hourly price series from 2023-04-01 through 2023-10-31,
  each with 5,136 unique hours;
- 12 official option gaps and two archives with no valid hour-0 row were recorded
  as missing/invalid and never filled.

Rows labelled hour 0 were conservatively treated as available at 01:00 UTC. Raw
archives, SHA-256 hashes and a request-level manifest are retained under
`data/external/binance_option_vol_front`.

## Frozen experiment verdicts

| Version | Candidate | Coverage | Primary result | Verdict |
|---|---|---:|---:|---|
| v16.7 | alt-vol front -> long BTC straddle | 1 trade | -20.56 bp/day | reject: event/tenor coverage failed |
| v16.8 | rich IV + quiet front -> short BTC straddle, 24h | 30 trades | -46.77 bp/day | reject: all periods negative |
| v16.9 | same short straddle, 7d + daily delta hedge | 11 trades | -79.91 bp/week | reject: holdout -304.21 bp/week |
| v17.0 | 25-delta skew level -> BTC direction | 29 trades | -7.51 bp/day | reject: validation -48.16 bp/day |
| v17.1 | 25-delta skew innovation -> BTC direction | 10 trades | -63.96 bp/day | reject: no holdout trades |

No candidate passed its preregistered gates. The positive reversed-innovation
diagnostic in v17.1 is only a ten-trade, no-holdout result discovered after the
frozen direction failed; it is not eligible to become a sign-flipped candidate.

## Mechanism conclusions

The alt-minus-BTC volatility front contains forecasting information but not
executable option alpha in this sample. Across 118 executable daily straddles,
front gap had Spearman correlation +0.33 with the next day's BTC realized-volatility
change. The highest front-gap quartile was followed by a mean 55.6% RV increase.
Yet front gap correlated only +0.04 with long-straddle net return, and all four
front quartiles lost after execution.

The option execution loss is real rather than a hedge-sign error. For all 118
daily pairs, long gross plus short gross equalled the negative entry-and-exit
bid/ask width with maximum error `3.47e-18`. Average two-sided option spread loss
was 63.42 bp of BTC notional, before approximately 12.75 bp of option/hedge fees.
Extending to seven days reduced neither the adverse option path nor short-gamma
tail risk enough to create a premium.

The surface also failed as a low-cost directional signal. Level skew was
period-degenerate (all development signals long and all validation signals short),
while skew innovation was sparse, unstable and wrong in the frozen demand-following
direction.

## Decision boundary

- Reject v16.7--v17.1; do not create candidate, shadow, PaperLive or leverage
  permissions from this round.
- Do not tune the event width, DTE band, skew threshold, holding period or v17.1
  direction on the same 2023 archive.
- Preserve the option dataset because it is an orthogonal research asset and an
  execution benchmark.
- The next defensible option-volatility action requires a longer executable
  surface with bid/ask and preferably intraday delta paths, or a forward recorder.
  A longer Deribit reconstruction is a data project, not permission to revisit
  these Binance thresholds.
- The continuous alt-volatility front may be retained as diagnostic telemetry for
  future volatility instruments, but the current evidence does not make it a
  tradable entry signal or a proven strategy overlay.

No remote host, PaperLive process, leverage setting or real-order permission was
changed in this round.
