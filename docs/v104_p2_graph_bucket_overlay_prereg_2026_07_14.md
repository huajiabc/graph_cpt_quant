# v10.4 P2 Graph-Bucket Return Overlay - Pre-Registration

## Scope

This is a historical P2 overlay audit motivated by the frozen v10.3 result.
It cannot change P2, PaperLive, shadow, canary, sizing, leverage, or live
permissions. Missing graph context fails open and preserves the P2 candidate.

## Frozen population and timing

- Candidate pool: deduplicated CIC1/CIC2 rows from the existing
  `paper_portfolio_trades.parquet`, preserving CIC1 priority and the frozen P2
  entry/exit outcomes.
- Portfolio: the existing first-come P2_EW maximum-eight selection logic.
- Graph features: v10.3 real monthly top-five correlation-neighbor bucket
  panel, latest `feature_time <= entry_time`, maximum staleness 15 minutes.
- Graph history begins in 2025-08; earlier P2 rows remain unfiltered.
- Development: 2025-08 through 2025-12; validation: 2026-01 through 2026-03;
  holdout: 2026-04 onward.

## Frozen overlay

`P2_AVOID_STRONG_BUCKET_LAGGARD` blocks a P2 candidate only when every field is
true at entry:

- bucket 1h return at least +0.50%;
- bucket-return cross-sectional percentile at least 80%;
- positive-neighbor breadth at least 60%;
- bucket excess versus market median at least +0.20%;
- bucket-minus-target 1h lag gap at least +0.30%.

The threshold is inherited unchanged from v10.3. Local 15m direction is not
used because v10.3 found both turn and no-turn laggards unattractive after
cost. All non-laggard and uncovered candidates remain eligible.

After filtering, the same max-eight P2_EW portfolio is re-simulated so later
candidates may fill released slots.

## Costs and controls

- `net_rt20`, `net_rt40`, and `net_rt60` mean total round-trip costs of 20, 40,
  and 60bp. `net_rt40` matches the historical P2 field formerly labelled
  `net_return_20bp` (20bp per side).
- 500 within-month joint permutations of all bucket fields across P2
  candidates, followed by a full overlay and max-eight re-simulation.
- One-day shifted real-bucket feature placebo.
- Baseline P2_EW with no graph filter.
- Entry-day block bootstrap of overlay-minus-baseline daily net_rt40.
- Report graph coverage, blocked-candidate outcomes, chronological splits,
  month/symbol concentration, selected sets, and freed-slot replacements.

## Frozen decision gates

At most `p2_graph_bucket_forward_watch_only` requires all of:

1. at least 120 graph-covered candidate rows and 100 overlay-selected trades;
2. at least 25 overlay-selected validation and 25 holdout trades;
3. overlay portfolio net_rt40 is positive full, validation, and holdout;
4. overlay-minus-baseline net_rt40 is positive in validation and holdout;
5. blocked laggard candidates have negative mean net_rt40;
6. full overlay lift exceeds the within-month permutation 90th percentile;
7. full overlay lift beats the one-day shifted placebo;
8. entry-day bootstrap 95% lower bound for lift is positive;
9. the overlay retains at least 70% of the deduplicated P2 candidate pool;
10. no month or symbol contributes more than 35% of positive overlay net_rt40.

Passing remains forward-watch only because the overlay was selected after
v10.3. Failure means continuous return-correlation bucket state does not
improve the frozen P2 portfolio under this specification.
