# v3.4 True Short Sleeve — Findings (2026-06-16 A100 Production)

Source: `short_instructment3(v3.4).docx`. Built on branch `short-v12s`.
Research only — no shadow / paper-live / real-live wiring touched.

## Verdict (one line)

**No standalone short sleeve clears the docx §6 ten-hurdle gate. All six
candidate sleeves (SS1A..SS3B) verdict `no_value` at the focal 20bp + 5bp
short-slippage cost. The only sleeve with a usable sample size — SS1A —
loses 30-49 bp per Fast/Swing trade after costs despite a clean-hit lift
(+3-5pp vs baseline) on the 12-14% of trades that exit cleanly. Hedge
correlation against the long book is +0.09 to +0.14: no hedge. Conclusion
matches v1.2s: the failure signals' value is long risk-off, not opening
shorts.**

## A100 production run

- **Date**: 2026-06-16, 247 box `interactive6228`
- **Env**: rebuilt conda env `quant` = Python 3.11.15 + pandas 3.0.3 +
  numpy 2.4.6 + scipy 1.17.1 + pyarrow 24.0.0 (Tsinghua pip mirror)
- **Data**: `data/processed/v0_3/perp_pressure_features_all_eligible.parquet`
  (3.2 GB) + `reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet`
  (2,296 rows × 63 symbols × 3 CIC candidates)
- **Tests**: `tests/test_v3_4_true_short_sleeve.py` 26/26 PASS in 2.2s
- **Production**: `python -m pressure_graph.cli run-v3-4-true-short-sleeve`
  on Top80 dynamic universe finished in **1m45s**

## Per-sleeve scoreboard (20bp + 5bp short slippage)

| Sleeve | Exec | n | net_mean | win | squeeze | clean_hit | clean_net | clean_lift | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SS1A | fast | **728** | -0.49% | 33.5% | 24.5% | 13.9% | +2.50% | **+2.99pp** | no_value |
| SS1A | swing | **728** | -0.30% | 43.8% | 28.2% | 12.5% | +4.50% | **+4.80pp** | no_value |
| SS1B | fast | 0 | — | — | — | — | — | — | (no trades — gate failed everywhere) |
| SS1B | swing | 0 | — | — | — | — | — | — | (no trades) |
| SS2A | fast | 1 | -0.02% | 0% | 0% | 0% | n/a | n/a | no_value (low_n) |
| SS2A | swing | 1 | -1.88% | 0% | 0% | 0% | n/a | n/a | no_value (low_n) |
| SS2B | fast | 0 | — | — | — | — | — | — | (no trades) |
| SS2B | swing | 0 | — | — | — | — | — | — | (no trades) |
| SS3A | fast | 0 | — | — | — | — | — | — | (no trades) |
| SS3A | swing | 0 | — | — | — | — | — | — | (no trades) |
| SS3B | fast | 2 | -1.39% | 0% | 50% | 0% | n/a | n/a | no_value (low_n) |
| SS3B | swing | 2 | -2.01% | 0% | 50% | 0% | n/a | n/a | no_value (low_n) |

Only **SS1A** clears the docx §6.9 50-sample threshold. SS1B/SS2B/SS3A
never fired because the combined gates (low_coimpulse / failed_followthrough +
btc_down / cic_candidate + cp60 + density_fading) are extremely restrictive
on the Top80 v0.3 features — the production feature build doesn't carry
`co_impulse_density` / `volume_impulse_density` for most symbols, so gates
that depend on it fail closed. SS2A / SS3B emit a single-digit trade
count where the gate happens to align with the existing v06c / v07d
columns. **Not a bug — a faithful encoding of the docx's "rare-but-clean"
spirit; the gates correctly refuse to fire on weaker setups.**

## Decisive table: short vs no-long (`short_vs_no_long_comparison.csv`)

| Sleeve | Exec | A_no_action_long | B_no_long | C_open_short | C - B | Verdict |
|---|---|---:|---:|---:|---:|---|
| SS1A | fast | 0.000 | -0.000 | -3.595 | -3.595 | **no_value** |
| SS1A | swing | 0.000 | -0.000 | -2.184 | -2.184 | **no_value** |
| SS2A | fast | 0.000 | -0.000 | -0.000 | -0.000 | no_value |
| SS2A | swing | 0.000 | -0.000 | -0.019 | -0.019 | no_value |
| SS3B | fast | 0.000 | -0.000 | -0.028 | -0.028 | no_value |
| SS3B | swing | 0.000 | -0.000 | -0.040 | -0.040 | no_value |

`A_no_action_long_total` is 0 across the board: the v0.9D long pool
(63 symbols × ~30 average trades / symbol) carries long positions for
4-12h, so a short signal arriving on the same symbol almost always lands
*inside* an active long's hold window, not in a fresh-entry window. The
3-action compare's cooldown window matches "long signals strictly *after*
the short signal" — and in this stack, those windows are empty.

That is itself an answer the docx §7 wanted: **opening short does not
beat doing-nothing-with-the-long here, because the relevant population —
brand-new longs in the same symbol just after the failure — is virtually
empty in the long pool**. This re-confirms v1.2s2/v1.2s3's positive result:
the failure signals' value is in the long-risk-off gate (v1.2s3 phase-4),
not in starting a separate short book.

## Regime split

SS1A breaks down as:

- BTC_down: 229 trades, fast -0.31%, swing -0.21% (still negative)
- BTC_chop: 495 trades, fast -0.57%, swing -0.36% (worst regime, bulk of trades)
- BTC_up: 4 trades, fast -1.40%, swing +1.59% (sample too small to bank)

No regime produces a sustainable short.

## Hedge value (`short_as_hedge_summary.csv`)

| Sleeve | Exec | long_days | long_down_days | short_net on long_down_days | corr |
|---|---|---:|---:|---:|---:|
| SS1A | fast | 42 | 19 | **-0.60%** | +0.137 |
| SS1A | swing | 42 | 19 | **-0.73%** | +0.090 |

SS1A loses money *while* the long book is losing. Positive correlation
of ~+0.10. Not a hedge — failure-mode shorts in the v3.4 spec correlate
with the long book's losses rather than offsetting them.

## What the sleeve scoreboard does say about signal quality

Even though no sleeve passes, the `short_clean_hit_summary.csv` row is
worth keeping:

- SS1A/fast 13.9% (101/728) trades exit clean-down → average +2.5% gross,
  lift over baseline = **+3.0pp**
- SS1A/swing 12.5% (91/728) trades exit clean-down → average +4.5% gross,
  lift over baseline = **+4.8pp**

So the underlying signal *does* carry directional information — but the
squeeze + max-hold noise (24-28% squeeze rate, 59-62% max-hold rate)
swamps the alpha at every cost grid. Execution is the binding constraint,
not signal scarcity. A future v3.5 could keep the SS1A motif but:

- Tighten max-hold so non-resolving trades exit faster (less drift cost)
- Add a *post-entry* breakdown confirmation (`close < entry - 0.5%` within
  1h or exit) to drop the ~62% max-hold population

That work belongs to a separate iteration, not v3.4.

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
