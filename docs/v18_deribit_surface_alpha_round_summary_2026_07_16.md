# v18 Deribit Surface Alpha Round Summary

Status: `DATA_BRANCH_ACCEPTED_ALPHA_BRANCH_REJECTED`

## What changed

- Added a reproducible archived Deribit BTC option-trade collector.
- Reconstructed inverse-option IV only from positive `volume` and `cost` bars;
  exchange-filled zero-volume prices are excluded.
- Added a 2021-03 to 2026-06 Binance USD-M 1h panel for BTC plus eleven liquid
  alts so option signals can be tested against multi-coin graph buckets.
- Built quarterly and monthly-expiry surface datasets, causal daily features,
  receiver graphs, random controls, timing placebos, sensitivities, and audits.

## Data result

- Quarterly archive: 272 queried contracts, 264 active contracts, 74,962 active
  option-trade hours, 567 quality daily surfaces across 22 expiries.
- Monthly archive: 786 queried contracts, 743 active contracts, 203,177 active
  option-trade hours, 1,638 quality date-expiry surfaces.
- Nearest-30DTE monthly series: 1,447 causal days across 63 expiries.
- Quarterly reconstructed ATM IV versus official BTC DVOL correlation: 0.9632.

## Alpha results

- v17.3 stress receiver short: 31 events, -37.70 bp/event after 20 bp cost;
  BTC-neutral version -47.24 bp. Both rejected.
- v17.4 stress receiver OCO: 26 events, -116.34 bp/event after filled-fraction
  costs; random-bucket percentile 0%. Rejected.
- v17.5 stress receiver-vs-insulator: -40.04 bp/event. Rejected.
- v17.5 relief receiver-vs-insulator initially showed +22.81 bp/event and 98.4%
  random-pair percentile, but had only five holdout events at -7.42 bp and a
  negative bootstrap lower bound. It did not pass.
- v17.6 monthly coverage extension removed that apparent lead: 73 relief events,
  -32.36 bp/event after cost, negative in every calendar year, random percentile
  50.8%. Rejected; no forward watch.

## Boundary update

The longer Deribit archive is useful as an orthogonal volatility research source,
but the tested skew level/innovation, directional receiver, OCO breakout, and
receiver-insulator transmission families do not provide deployable alpha. No
strategy, application, PaperLive, leverage, remote, or real-order permission was
created or changed. The pre-existing v16.5 fixed FSS3/TG1 candidate remains
unaffected.
