# v10.3 Graph Bucket-Return Diffusion - Pre-Registration

## Scope and permission

This is an offline graph-native alpha test. It cannot change P2, PaperLive,
shadow, canary, leverage, sizing, or live permissions.

The hypothesis is deliberately different from prior NIR/MIR work. Prior work
tested neighbor event counts, market impulse density, and individual directed
leaders. v10.3 tests the continuous return of a target's as-of neighbor bucket
and the target's subsequent catch-up relative to that bucket.

## Frozen graph and chronology

- Price grid: Bybit 15-minute feature table from 2025-06 through 2026-06.
- Graph: the existing monthly `return_corr_30d` top-five neighbor edges from
  `reports/v0_7b_neighbor_graph/neighbor_graph_edges.csv`.
- Each month's edges were estimated only from the trailing 30 days ending
  strictly before the month boundary.
- A target requires at least three contemporaneously covered neighbors.
- All graph-return features use bars closed by `feature_time`.
- Entry is the next 15-minute bar open at `feature_time`.
- Candidate state is sampled only on its false-to-true transition, followed by
  a four-hour target-symbol cooldown.

Chronological segments:

- development: 2025-08 through 2025-12;
- validation: 2026-01 through 2026-03;
- holdout: 2026-04 through the end of available history.

No threshold may change after candidate returns are read.

## Frozen bucket features

For each target and feature time, excluding the target itself:

- equal-weight neighbor-bucket returns over 15m, 1h, and 4h;
- positive-neighbor breadth over 1h;
- cross-neighbor 1h return dispersion;
- bucket 1h return percentile across covered targets at the same time;
- cross-sectional market-median 1h return;
- bucket excess return versus that market median;
- target lag gap: bucket 1h return minus target 1h return.

Future labels are target 4h/12h return and equal-weight future neighbor-bucket
4h/12h return, both unavailable to candidate construction.

## Frozen states

`GBR1_BROAD_LAG_CATCHUP` is primary:

- bucket 1h return at least +0.50%;
- bucket 1h cross-sectional percentile at least 80%;
- at least 60% of covered neighbors have positive 1h returns;
- bucket excess versus market median at least +0.20%;
- target lag gap at least +0.30%;
- target latest 15m return is positive.

`GBR2_LAG_NO_TURN` is a local-confirmation control with the same bucket and lag
conditions but target 15m return non-positive.

`GBR3_COIMPULSE_CONTINUATION` is a continuation comparator:

- the same bucket-return, percentile, breadth, and excess conditions;
- target 1h return is positive and no more than 0.10% below the bucket return;
- target latest 15m return is positive.

The three states are a frozen family. GBR2 and GBR3 cannot rescue a failed
GBR1 without familywise random-graph control.

## Portfolio and cost views

At each feature time, select at most three targets per state, ranked by target
lag gap for GBR1/GBR2 and target 1h return for GBR3. Equal-weight selected
targets form one timestamp portfolio observation.

- Raw-long gross: future target return.
- Catch-up gross: future target return minus the future equal-weight neighbor
  bucket return.
- Raw-long round-trip costs: 20, 30, and 50bp.
- Catch-up two-leg total round-trip costs: 40, 60, and 100bp.
- Primary horizon and outcome: 4h catch-up net40.
- 12h is secondary and cannot rescue a failed 4h result.

Cost labels always mean total round-trip basis points, removing the previous
single-side/round-trip ambiguity.

## Frozen controls

- 50 density-matched random graphs per month. Every target keeps the same
  neighbor count, and neighbors are sampled from that month's covered graph
  universe excluding the target.
- Family maximum across the three states for random-graph comparison.
- A one-day shifted real-bucket feature placebo, retaining the current target
  state and current future outcome.
- The real non-neighbor market-median return as an attribution field.
- Day-block bootstrap with 2,000 iterations.
- Monthly, target-symbol, and event-day contribution concentration.

## Frozen decision gates

At most `graph_bucket_research_candidate_only` requires all of:

1. at least 200 full, 60 validation, and 60 holdout timestamp portfolios;
2. at least ten target symbols, eight active months, and 30 active days;
3. validation and holdout 4h catch-up net40 are positive;
4. validation and holdout raw-long net20 are positive;
5. full 4h catch-up net60 is positive;
6. day-block bootstrap 95% lower bound for catch-up net40 is positive;
7. real catch-up net40 exceeds the random-graph family 90th percentile;
8. real catch-up net40 beats the one-day shifted-bucket placebo;
9. no single month, target symbol, or event day contributes more than 35% of
   positive catch-up net40;
10. five chronological full-sample buckets have non-negative mean catch-up
    net40.

Passing remains research-only because the graph and thresholds are historical.
Failure of all states closes this fixed monthly top-five correlation-bucket
specification, not every possible graph construction.
