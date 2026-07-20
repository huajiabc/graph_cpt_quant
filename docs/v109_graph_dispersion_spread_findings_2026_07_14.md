# v10.9 Residual Graph-Community Dispersion Spread Findings

Date: 2026-07-14

Verdict: `reject_graph_dispersion_family`.

## What was tested

Monthly BTC-residual correlation graphs were reduced to maximum spanning trees and cut into
eight frozen components. When a component's current cross-sectional residual-return dispersion
crossed its historical 95th percentile, the strategy traded a normalized top-third versus
bottom-third four-hour spread. Convergence and continuation directions were tested as one
family against 50 random partitions preserving component sizes and a one-day shifted signal.

## Results

| Candidate | Observations | Gross | Net20 | Validation net20 | Holdout net20 | Random-family percentile |
|---|---:|---:|---:|---:|---:|---:|
| Community convergence | 433 | -2.28 bp | -22.28 bp | -25.19 bp | -21.73 bp | 0% |
| Community continuation | 433 | +2.28 bp | -17.72 bp | -14.81 bp | -18.27 bp | 84% |

The continuation sign was better than convergence, but its gross edge was only 2.28 bp and
every five chronological net20 slice was negative. Its bootstrap interval was entirely negative
[-22.15 bp, -13.77 bp]. It beat the one-day shifted signal but did not clear the random-family
90th-percentile gate.

## Structural audit

The maximum-spanning-tree cut did not produce balanced local communities. In every month one
component contained 63-64 of the 72 non-BTC symbols; the remaining components were mostly
singletons. Only one component per month met the six-symbol trading minimum. Economically,
v10.9 was therefore close to a broad-universe extreme-dispersion spread rather than evidence
that local graph communities add alpha.

This is a valid rejection of the preregistered MST-cut construction. It is not a reason to stop
graph research, but it prevents treating the 84th-percentile continuation result as a community
alpha.

## Remaining graph-alpha space

The next distinct hypotheses are:

1. balanced spectral partitions with explicit size constraints, evaluated as a new version and
   not used to rescue v10.9;
2. changes in topology—centrality, community migration, and correlation breakdown—rather than
   returns conditional on static edges;
3. the synchronized cross-venue aggressor-flow graph after v10.7 reaches its forward-data gate;
4. graph-aware allocation of already independent alpha sleeves, where the objective is drawdown
   and concentration reduction rather than a new entry signal.

No PaperLive, leverage, or live permission changed.
