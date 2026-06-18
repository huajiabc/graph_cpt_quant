# Candidate Verdict Schema

> Source of truth for the `validation/gate_checks.py` module. Per
> 77.docx P0 (`vEval Global Gate5 / Concentration Validator`), every
> report that emits a candidate verdict must conform to this surface.

## Why this exists

The v7S work surfaced a load-bearing methodology bug: the original
`_evaluate_gates` in `v7s_short_alpha.py` implemented gate5 as
`month_capped_net > 0`, which was a softer test than the closure doc's
reopen §1 wording ("no single month contributes ≥ 35 % of the alpha").
A1_imb10bp h24 verdicted PROMOTE under that soft gate while one month
(2025-10) carried 63.7 % of the alpha. Walk-forward over disjoint
thirds showed bucket 0 (2025-07 → 10) was NEGATIVE. Bootstrap 95 % CI
on N=56 straddled zero.

Per the 77.docx P0 directive, the fix is global: any report that
imports `validation.gate_checks` inherits the tightened gates. Ad-hoc
per-report gate code is no longer the source of truth.

## Required output fields per candidate (CSV column names)

Distribution / concentration metrics:

| field | meaning |
|-------|---------|
| `n_trades` | number of executions in the candidate's sample |
| `total_net` | sum of `net20` across the sample |
| `best_month_share` | signed share of the largest-net month vs total |
| `month_cap35_net` | sum of per-month nets capped at 35 % of total |
| `leave_one_month_min` | minimum total after removing each month in turn |
| `best_symbol_share` | signed share of the largest-net symbol vs total |
| `leave_one_symbol_min` | minimum total after removing each symbol in turn |
| `months` | number of distinct months in the sample |
| `symbols` | number of distinct symbols in the sample |

Robustness metrics:

| field | meaning |
|-------|---------|
| `bootstrap_mean` | mean of `net20` |
| `bootstrap_ci_lo` | 2.5th percentile of 5000-resample means |
| `bootstrap_ci_hi` | 97.5th percentile of 5000-resample means |
| `bootstrap_p_positive` | fraction of resample means strictly above zero |
| `walk_forward_min_net` | minimum bucket mean_net20 over 3 disjoint thirds |
| `walk_forward_max_net` | maximum bucket mean_net20 |
| `walk_forward_delta` | `walk_forward_max_net - walk_forward_min_net` |
| `random_p50` | 50th percentile of 1000-resample means from the pool |
| `random_p75` | 75th percentile |
| `random_p90` | 90th percentile |
| `candidate_mean_vs_random` | `bootstrap_mean - random_p50` |

Gate pass booleans:

| field | pass condition |
|-------|----------------|
| `gate_best_month_share_pass` | `best_month_share ≤ month_cap_pct` (default 0.35) |
| `gate_month_cap_pass` | `month_cap35_net > 0` |
| `gate_leave_one_month_pass` | `leave_one_month_min > 0` |
| `gate_best_symbol_share_pass` | `best_symbol_share ≤ symbol_share_max` (default 0.35) |
| `gate_leave_one_symbol_pass` | `leave_one_symbol_min > 0` |
| `gate_random_pass` | `bootstrap_mean ≥ random_p90` (default percentile) |
| `gate_bootstrap_pass` | `bootstrap_ci_lo > 0` |
| `gate_walk_forward_pass` | `walk_forward_min_net > 0` |

Plus failure trail and verdict:

| field | meaning |
|-------|---------|
| `gate_failures` | `;`-joined list of failing gate names |
| `final_verdict` | one of `promote` / `risk_off_only` / `diagnostic_only` / `no_value` / `no_data` |

## Verdict ladder

The five-way verdict is resolved AFTER all gates are computed. The
distribution / concentration gates dominate — any failure there short-
circuits to `no_value`. Walk-forward sits as a soft demote to
`diagnostic_only` (the candidate is interesting research output but
not tradable). Bootstrap + random baseline determine whether a
distribution-clean cell graduates to `promote`, `risk_off_only`, or
back to `no_value`.

```
if no rows                                                       → no_data
elif any distribution / concentration gate fails                  → no_value
elif walk_forward_min_net ≤ 0                                     → diagnostic_only
elif bootstrap_ci_lo > 0 AND candidate_mean ≥ random_p90          → promote
elif bootstrap_ci_lo > 0 OR  candidate_mean ≥ random_p90          → risk_off_only
else                                                              → no_value
```

`risk_off_only` is the closest match to v6S Path C's historical
verdict: discipline passes broadly but ONE of (significance,
random-baseline beat) fails. An operator may take it under risk-off
discretion; the framework never auto-triggers.

`diagnostic_only` is new in this schema — a candidate that survives
every concentration check but loses money in at least one disjoint
walk-forward bucket. The v7S A1_imb10bp h24 cell would land here under
the tightened framework if its month concentration had cleared.

## Domain-specific gate injection

Report modules carry domain-specific gates that don't generalise (cost
stress at 30 bp, squeeze share, vs-no-long head-to-head, clean-short
hit rate). They inject these via `additional_failures` /
`additional_passes` on `evaluate_candidate_verdict` so the verdict
surface stays uniform:

```python
verdict = evaluate_candidate_verdict(
    trades,
    additional_failures=tuple(
        name for name in ("gate3_clean_hit", "gate4_squeeze", "gate9_short_beats_no_long")
        if not domain_specific_pass(name)
    ),
)
```

A caller-injected failure short-circuits the verdict to `no_value` even
if all distribution gates pass — matching the existing v7S behaviour.

## Calling conventions

Required columns in the trades frame:

- `net20` (or another net column passed via `net_col`)
- `signal_time` (or another time column passed via `time_col`)
- `month` — string in `YYYY-MM` format
- `symbol` — string

Optional `pool_nets` argument for random baseline — defaults to the
candidate's own `net20` (matched bootstrap). Pass an external pool
(e.g. all-direction concatenated `net20` array) to compute a
cross-candidate random baseline.

## Thresholds

All thresholds live on `GateThresholds`. Defaults match the closure
doc's reopen criteria + the v7S validation findings:

- `month_cap_pct = 0.35` (closure §1)
- `symbol_share_max = 0.35`
- `bootstrap_draws = 5000`, `bootstrap_seed = 20260617`
- `walk_forward_buckets = 3`
- `random_draws = 1000`, `random_seed = 20260618`
- `random_pass_percentile = 90` (candidate beats random p90)

Pass a non-default `GateThresholds()` to relax for diagnostic
candidates; do NOT relax for promotable candidates.

## Adoption checklist

A report that adopts this schema must:

1. Import `evaluate_candidate_verdict` from `pressure_graph.validation`.
2. Group its trades by `(direction, candidate_code, execution)` (or
   the equivalent partitioning).
3. Call `evaluate_candidate_verdict(sub_trades, …)` per partition.
4. Flatten via `verdict.to_dict()` and write to
   `<report>/short_candidate_summary.csv` (or the report's analog).
5. Make any domain-specific gates be `additional_failures` so they
   appear in `gate_failures` alongside the canonical ones.

See `src/pressure_graph/reports/v7s_short_alpha.py` for the migration
target (the v7S evaluator will be refactored on top of this module).
