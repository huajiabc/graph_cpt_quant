# v3.4 True Short Sleeve — Framework + Pending A100 Run

Source: `short_instructment3(v3.4).docx`. Built on branch `short-v12s`.
Research only — no shadow / paper-live / real-live wiring touched.

## Verdict (provisional)

**Framework landed; the 6 candidate sleeves (SS1A..SS3B) and the decisive
3-action comparison (no-action / no-long cooldown / open-short) compile
and unit-test green locally. The production verdict — whether any sleeve
clears the docx §6 ten-hurdle gate at 20bp + 5bp short slippage — depends
on the A100 run against the v0.9D capacity trade cache + the v0.3 feature
parquet. This document will be updated once that run produces the 9
docx-mandated CSVs.**

## What was built

### Six candidate sleeves

| Code | Sleeve | Path |
|---|---|---|
| SS1A | Failed Reclaim Breakdown + BTC_not_up | S1/S3/S5 motif → failed reclaim → close < reclaim_low → BTC not strong |
| SS1B | Failed Reclaim Breakdown + low co-impulse | Same path, gate replaced with low_coimpulse |
| SS2A | Crowded Long Stall + BTC_down | S3 motif → price_stall → failed_followthrough → BTC_down |
| SS2B | Crowded Long Stall + low co-impulse | Same path with low_coimpulse instead of BTC_down |
| SS3A | CIC Failure + density fading | CIC candidate → CP60_would_exit → breakdown below entry → density fading |
| SS3B | CIC Failure + no Protect_A | CIC candidate → CP60_would_exit → breakdown below pullback low → no Protect_A |

All sleeves wait for the breakdown bar (close < reference low) — never short
at the failure event itself. That is the v3.4 docx point: a short here is
the reverse of CP60, not "shorting strong names from the top".

### Fixed short execution

- Fast short: `tp_down=3%`, `sl_up=2%`, `max_hold=4h` (16 × 15m bars)
- Swing short: `tp_down=5%`, `sl_up=3%`, `max_hold=12h` (48 × 15m bars)
- Cost grid: 10/20/30/50 bp + extra short slippage 5/10 bp
- Focal cell (`docx §6.1`): 20 bp + 5 bp short slippage

### v3.4 label extension (`labels/short.py`)

Added on top of the existing v1.2s label family:

- 24h window with `hit_down_8pct_24h` and the matching squeeze probe
- `short_first_touch_<window>` ∈ {`DOWN_FIRST`, `UP_FIRST`, `NEITHER`}
- `time_to_down_hit_<window>` and `time_to_up_stop_<window>` (bar offsets)
- `clean_short_hit_<window>` = target hit AND not squeezed first (docx §1)

Existing tests (`test_short_labels_execution.py`) continue to pass — the
default windows now include 24h but tests that pass narrower dicts still
build the labels they used to.

### Three-action comparison (`_three_action_compare`)

For every short signal at (symbol, signal_time), the orchestrator joins
the v0.9D capacity trade cache for longs in (signal_time, signal_time + cd]
and computes:

| action | what it measures |
|---|---|
| A no_action | Realized long net20 of any long that *would have* entered (signal stays passive) |
| B no_long_cooldown | -A (avoided loss / sacrificed gain — same magnitude flipped) |
| C open_short | Short net20 from Fast / Swing execution at the focal cost grid |

The decisive row in `short_vs_no_long_comparison.csv` is `C_minus_B`:

- `C_minus_B > 0` *and* `C_open_short_total > 0` → `true_short_value`
- otherwise if `B_no_long_cooldown_total > 0` → `risk_off_value`
- otherwise → `no_value`

This is the docx §7 gate stated programmatically: if no-long cooldown beats
opening short, the signal stays as risk-off (the v1.2s2 / v1.2s3 product),
not a short.

### Hedge value report (`_short_as_hedge`)

`short_as_hedge_summary.csv` cross-tabulates per-sleeve daily PnL against
the long book's daily PnL, isolating:

- `short_net_on_long_down_days` — how the sleeve behaves when the long book
  is losing
- `short_net_on_long_up_days` — the upside drag when the long book is winning
- `corr_long_short` — Pearson correlation across days

Per docx §9, hedge value ≠ standalone alpha; any sleeve that survives only
the hedge column is treated as a small sleeve.

## Outputs (9 CSVs + notes)

Under `reports/v3_4_true_short_sleeve/`:

- `short_sleeve_trades.csv` — raw per-signal Fast/Swing rows
- `short_candidate_summary.csv` — cost-grid × sleeve × execution scoreboard
- `short_first_touch_summary.csv` — tp / stop / max_hold / squeeze mix
- `short_clean_hit_summary.csv` — docx §1 clean_short_hit cut
- `short_cost_stress.csv` — net per cost-cell
- `short_regime_split.csv` — BTC_down / BTC_chop / BTC_up split
- `short_month_symbol_attribution.csv` — month × symbol PnL grid
- `short_vs_no_long_comparison.csv` — **decisive**: per-sleeve A / B / C
  totals + `C_minus_B` + `verdict`
- `short_as_hedge_summary.csv` — hedge correlation against the long book
- `candidate_notes.md` — verdict synthesis stamped from the above

## How to run

Locally with synthetic features (smoke test of the pipeline shape):

```text
python -m pressure_graph.cli run-v3-4-true-short-sleeve
```

A100 production: ssh to the GPU box and run the same command with
`A100_PASSWORD` exported (see `scripts/_a100_run.py`).

## Discipline

- Every short entry waits for the breakdown bar — never the failure event.
- 3-action compare is the gate: `C - B > 0` AND `C > 0` AND `n ≥ 50`.
- Hedge value is reported separately; hedge-only sleeves stay tiny (docx §9).
- No paper-live / real-live permission changes.
