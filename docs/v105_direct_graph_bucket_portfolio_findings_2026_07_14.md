# v10.5 Direct Graph-Bucket Portfolio Findings

Date: 2026-07-14

Verdict: `reject_direct_graph_bucket_family`.

## What was tested

Each source node's monthly frozen top-five correlation neighbors were treated as an equal-weight
tradable sleeve. This is the most direct multi-coin formulation: forecast and trade the bucket,
not a target coin. At most three sleeves were combined at each false-to-true transition.

The preregistered family tested four-hour broad-bucket continuation and a short reversal after
the bucket's 15-minute return stopped being positive. Total round-trip costs were 20/30/50 bp.
The real graph was compared with 50 random-neighbor graphs and a one-day shifted signal.

## Four-hour results

| Candidate | Observations | Gross | Net20 | Validation net20 | Holdout net20 | Random-family percentile |
|---|---:|---:|---:|---:|---:|---:|
| Long broad-bucket continuation | 6,154 | +0.07 bp | -19.93 bp | -19.90 bp | -14.82 bp | 0% |
| Short bucket-turn reversal | 2,451 | +1.91 bp | -18.09 bp | -19.42 bp | -21.90 bp | 34% |

Continuation's bootstrap 95% interval was [-29.96 bp, -9.84 bp]. Reversal's interval was
[-28.82 bp, -6.55 bp]. Neither candidate produced a positive month distribution after costs.

## Secondary 12-hour check

- Continuation: full net20 -36.90 bp; validation -23.47 bp; holdout -16.70 bp.
- Reversal: full net20 -1.91 bp. Its development result was +7.09 bp, but validation was
  -10.60 bp and holdout -14.50 bp.

The 12-hour result therefore does not rescue either mechanism.

## Interpretation

- Strong one-hour multi-coin movement is contemporaneous information, not a four-hour bucket
  return forecast in this construction.
- Trading the whole bucket removes target-selection noise but does not reveal hidden gross edge;
  gross returns are approximately zero and far below plausible execution cost.
- The real correlation graph is not superior to random membership. Fixed top-five return
  correlation should not be treated as an alpha-generating graph.

The next graph research should change the graph's information content rather than retune these
entry thresholds: directed lead-lag edges, residual/partial-correlation edges, or slow community
regime exposures are distinct hypotheses. Re-running fixed-correlation bucket momentum,
catch-up, or P2 blocking would be specification search against a decisively negative result.

No PaperLive, leverage, or live permission changed.
