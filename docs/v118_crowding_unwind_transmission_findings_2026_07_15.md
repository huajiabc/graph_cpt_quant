# v11.8 Crowding-Unwind Transmission Findings

Date: 2026-07-15

Verdict: `reject_crowding_unwind_community_alpha`. No PaperLive, live, sizing,
or leverage permission changed.

## Data quality

The newly acquired Bybit account-ratio archive contains 8,761 hourly rows for
each of 73 symbols. The formal test retained 11 monthly contexts with exactly
72 non-BTC symbols in eight communities of nine. Crowding-z coverage was 100%
for every retained month.

## Result

| Candidate | Full n | Gross | Net20 | Validation net20 | Holdout net20 | Residual net40 | Random-partition percentile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Crowded-long unwind, short followers | 395 | +3.49 bp | -16.51 bp | -9.80 bp | -34.95 bp | -43.69 bp | 84% |
| Crowded-short squeeze, long followers | 32 | +20.47 bp | +0.47 bp | -39.10 bp | +5.83 bp | -14.95 bp | 30% |

The long-unwind bootstrap interval for net20 was entirely negative (-49.72 bp
to -3.95 bp). It worsened across time: four of five chronological fifths were
negative and the final fifth was -39.07 bp.

The short-squeeze branch approximately covered the focal cost in full, but it
had only 32 observations, negative validation, negative residual expectancy,
and lost to random communities. Its positive months were also concentrated:
one month contributed 48.73% of positive payoff.

## Interpretation

Account-count crowding plus falling OI identifies stressed hours, but the
frozen price communities do not identify who absorbs the unwind next. This is
another case where a market-wide volatility state is real while the graph
attribution is not tradeable.
