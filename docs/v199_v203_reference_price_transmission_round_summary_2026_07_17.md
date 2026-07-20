# v19.9-v20.3 Reference-Price Transmission Alpha Round Summary

## Data and identity

The checksummed Binance USD-M mark-price and index-price collections each cover
46 symbols, 1,497,024 rows, and 690 verified archives with no missing archives.
The research overlap contains 1,493,114 complete symbol-bars. The independent
v19.9 audit passed 16/16 checks.

`mark / index - 1` and the official premium-index close are related but not
identical: aggregate correlation is 0.9597, median absolute difference is
0.000132, and the 99th percentile difference is 0.001509. Treating both as
independent alpha without orthogonalization would double count information.

## Double-orthogonal feature construction

v20.0 removed premium innovation twice: first with a shifted prior-30-day
per-symbol rolling regression, then with a contemporaneous cross-sectional
regression at the completed feature bar. The resulting reference-lag score has
median/max absolute cross-sectional premium correlation of 5.65e-17/3.07e-15.
Its median/max absolute per-symbol time-series correlation is 0.0345/0.1276.

Four frozen feature configurations had adequate pre-reveal coverage:

| Configuration | Bucket events | Development | Validation | Holdout |
|---|---:|---:|---:|---:|
| Global reference-lag catch-up | 986 | 508 | 211 | 267 |
| Global trade-overshoot fade | 619 | 297 | 158 | 164 |
| Community reference-lag catch-up | 227 | 74 | 74 | 79 |
| Community trade-overshoot fade | 360 | 179 | 98 | 83 |

## Preregistered v20.1 reveal

All four beta-neutral candidates were rejected. Independent repricing passed
22/22 checks.

| Candidate | Events | Gross bp | Primary bp | Delayed bp | Random family percentile |
|---|---:|---:|---:|---:|---:|
| Global reference catch-up | 951 | 0.596 | -19.404 | -18.827 | 0.002 |
| Global trade overshoot fade | 597 | 0.165 | -19.835 | -20.872 | 0.000 |
| Community reference catch-up | 227 | -0.136 | -20.136 | -15.323 | 0.000 |
| Community trade overshoot fade | 360 | 1.601 | -18.399 | -23.507 | 0.506 |

The gross edge is too small even before debating the 20 bp primary round-trip
cost. This is a mechanism failure for tradable beta-neutral transmission, not
merely a conservative-cost rejection.

## Posthoc community peer hedge

v20.1 attribution showed that the community trade-overshoot alt sleeve was
positive while its BTC hedge was negative. A feature-only follow-up retained
237 events with at least two unselected same-community peers. The posthoc v20.3
book faded selected overshoot receivers and used those peers as the opposite
0.5-notional sleeve. It was also rejected; independent repricing passed 18/18.

| Scope | Selected contribution bp | Peer contribution bp | Gross bp | Primary bp |
|---|---:|---:|---:|---:|
| Development | 9.801 | -10.907 | -1.106 | -21.106 |
| Validation | 10.908 | -10.098 | 0.811 | -19.189 |
| Holdout | 2.692 | -2.442 | 0.250 | -19.750 |
| Full | 7.977 | -8.205 | -0.228 | -20.228 |

The selected-only directional fade has 15.953 bp full-sample gross return, but
falls from 19.601/21.817 bp in development/validation to 5.384 bp in holdout.
That observation is posthoc, below the frozen 20 bp cost, and not permission to
remove the hedge or add leverage.

## Research conclusion and next branch

Same-bar reference and execution-price dislocations identify synchronous
community reversal, not an internal relative-value spread. BTC and same-community
peer hedges both remove nearly all gross effect. Repeating hedge substitutions
on this event family is therefore low-value.

The next non-duplicative branch is delayed volatility relay:

1. estimate monthly directed leader-to-receiver edges from only the preceding
   30 days of index returns and absolute returns;
2. require a current multi-leader community shock;
3. exclude leaders and already-reacted receivers;
4. rank only historically delayed receivers by frozen edge strength and current
   response deficit; and
5. freeze coverage before revealing next-bar catch-up or reversal returns.

No live, PaperLive, application, leverage, remote, or order state changed in
this round.
