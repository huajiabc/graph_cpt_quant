# v6S Path C Short Validation — findings

**Status: research only.** A100 production run finished against the v0.9D
capacity trade cache (top-30 universe, conda quant py3.11). v6S confirms what
instructment5 §5 demanded: the seven discipline checks make or break Path C.

**Verdict: v6S DEMOTES the v4S +1.52% headline to `risk_off_only` (an opt-in
discretionary action) — NOT a default sleeve.** Two criteria killed it:

1. **Month stability fails hard** — best-month contribution = **85.06%**.
   One month carries almost the entire +1.54% mean. With N=63 and a single
   month doing 85% of the work, the v4S result is regime-specific, not a
   stable alpha.
2. **Hedge value is negative-or-flat** — Pearson corr vs long-stack monthly
   net = **−0.167** (weak, not the ≤ −0.3 we required); and in the long
   stack's worst month, S-C1 short *also* lost **−5.71%**. When the long
   book needs help most, this short hurts instead of helping.

Three criteria pass (min_samples, beats_no_long, cost_stress_robust); two
more pass (symbol_stability, clean_short_squeeze_ok). Five out of seven.
Per instructment5 §5 the threshold for `ship_to_shadow` is **all seven**, so
the candidates land in `risk_off_only` — usable as a flagged discretionary
action only, not the default behaviour.

## Setup

- Event source: v4S Path C detector (crowded stall + BTC weakness + failed
  follow-through) — identical to the v4S run, **63 events** across the
  top-30 universe (Swing execution only).
- Four candidates:
  - `S_C1_normal_short_swing` — 1.0× short, Swing exit
  - `S_C2_small_short_swing` — 0.5× short, Swing exit
  - `S_C3a_no_long` — passive: forward-window long blocked (12h)
  - `S_C3b_no_overflow` — passive: only O6-overflow long blocked
- Focal cost: 20 bps + 0 bps slippage + 30% APR funding accrual (shorts
  receive at funding_percentile ≥ 70).
- Cost-stress grid: 10/20/30/50 bps × 0/5/10 bps slippage = 12 cost cells.
- Stability thresholds: month_cap_pct 35%, min_samples 20.
- Hedge correlation against the v0.9D long-stack monthly net.

## Per-candidate

| candidate | N | mean | win% | vs B_no_long | month % | sym % | hedge ρ | worst-long-month PnL | verdict |
|-----------|---|------|------|--------------|---------|-------|---------|----------------------|---------|
| S-C1 normal short | 63 | **+1.54%** | 62% | +1.54% | **85.06%** | 13.70% | **−0.167** | **−5.71%** | risk_off_only |
| S-C2 small short | 63 | +0.77% | 62% | +0.77% | **85.06%** | 13.70% | **−0.167** | **−2.86%** | risk_off_only |
| S-C3a no_long | 63 | 0.00% | 0% | 0.00% | n/a | n/a | n/a | 0.00% | reject (degenerate) |
| S-C3b no_overflow | 63 | 0.00% | 0% | 0.00% | n/a | n/a | n/a | 0.00% | reject (degenerate) |

The S-C3 candidates are degenerate because **no v0.9D long entered within the
12h forward window of any Path C event** — Path C fires under "crowded
funding + BTC weakness" and the CIC long stack already stays out of those
regimes. So the `no_long` control is not "blocking longs that would have
happened" — it's "blocking nothing because nothing was about to happen".
That's a finding in itself: the long stack already does the right risk-off
on Path C symbols, the short is *additive* exposure, not a substitute.

## Cost-stress grid (S-C1 mean_net, swing)

| cost bps \ slip bps | 0 | 5 | 10 |
|---------------------|---|---|----|
| 10 | +1.74% | +1.64% | +1.54% |
| 20 | +1.54% | +1.44% | +1.34% |
| 30 | +1.34% | +1.24% | +1.14% |
| 50 | +0.94% | +0.84% | +0.74% |

Cost-stress is robust (positive across the full grid). The fragility is NOT
in costs — it's in time concentration.

## Clean-short labels (Swing exit)

| label | rate |
|-------|------|
| hit_down_3pct | 50.79% |
| hit_down_5pct | 49.21% |
| up_before_down_2pct | 41.27% |
| up_before_down_3pct | 23.81% |
| short_squeeze_before_hit | **15.87%** |
| mean_max_adverse_up | 2.77% |

The 15.87% squeeze rate clears the ≤ 20% threshold. The high `up_before_down_2pct`
(41%) is a warning: even when the direction eventually resolves down, you
get pushed against a 2%+ rally first ~40% of the time. With max_adverse_up
mean 2.77%, the squeeze is uncomfortable but survivable under the Swing
TP/SL band (5%/3%).

## Hedge correlation

- Pearson correlation across overlapping months: **−0.167** (weak negative).
- Worst long-stack month: long stack lost; S-C1 short ALSO lost **−5.71%**.
  Short does not hedge the long stack; it goes the wrong way when long
  stack hurts most.

This is the most important finding: v4S's promise of "another row in the
portfolio with low correlation" is **not** delivered by Path C. The
correlation is too weak to count as a hedge, and the worst-month overlap is
negative for the candidate (i.e. wrong sign).

## What v6S settles

- The v4S "Path C swing-short is alpha" claim does **not** survive
  discipline. One month drives 85% of the +1.54%. Restated as
  `risk_off_only` — an opt-in discretionary action, never the default.
- S-C3 (`no_long` / `no_overflow`) is empty because the long stack already
  stays out of crowded-funding + BTC-weakness regimes. Path C short is
  *additive*, not *substitutional*.
- v6S confirms instructment5's prediction: **failure is long-stack risk
  context, not a short signal**. v3.5 F5+F3 remain the right shadow
  candidates; Path C does NOT graduate.

## Implications for the live stack

- Do not ship Path C swing-short as an automated sleeve.
- Path C can still inform discretionary risk-off — when the combo gate
  fires under BTC weakness, the operator may choose to open a small
  flagged short. But this is a manual decision under a labelled risk
  context, not an automated action.
- Future short-side work should look for signals with broader month
  coverage AND positive worst-month overlap with the long stack. The
  current short framework consistently fails one or the other.

## Provenance

- Code: `src/pressure_graph/reports/v6s_path_c_short_validation.py`
- CLI: `pressure-graph run-v6s-path-c-short-validation --config configs/v0_3.yaml`
- Pipeline: `pipeline.run_v6s_path_c_short_validation_from_features`
- Tests: `tests/test_v6s_path_c_short_validation.py` (13 cases, all green
  locally + on A100 conda `quant` env).
- Reports: `reports/v6s_path_c_short_validation/` — 7 CSVs + candidate_notes.md.
- Related: [[v4s-failure-state-graph]] (where the +1.52% headline came from),
  [[v3-5-failure-risk-layer-bridge]] (F5+F3 risk-layer wins that DO ship),
  [[v3-3-v3-4-failure-search-and-true-short]] (the standalone-short-is-dead lineage).
- A100 access: [[a100-ssh-access]].
