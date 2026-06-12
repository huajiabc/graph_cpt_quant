# v1.2s2 Long Risk-Off Overlay — Findings (2026-06-12)

Phase-2 of the short-side research and its actual product. v1.2s proved the
failure motifs are not tradeable as shorts but that their value (doc §8) is
*cutting long exposure*. This measures that directly: use S1/S3/S5 confirmations
as a risk-off gate on the CIC P2 long basket and compare against the un-gated book.

Tier: **research only** — no shadow / paper-live / real-live wiring.

## Verdict (one line)

**Standing aside on a name right after its own failure motif (symbol-level
risk-off) improves the long book on every axis at every capacity — net up,
drawdown ~30% smaller, return/drawdown +50–75%, monthly concentration down. The
market-breadth gate is too blunt and hurts net. This is where the short research
pays off: as a long risk-off gate, not a short book.**

## Setup

- Long book: the CIC P2 basket (CIC1+CIC2) from the v0.9D capacity trade cache,
  selected first-come at max 5/8/10 concurrent, 20bp.
- Gate signal: 10,216 S1/S3/S5 failure confirmations streamed from the same
  per-symbol feature path (S2 excluded — mistimed in v1.2s).
- Two gate modes, both strict as-of (a long is only gated by confirmations whose
  feature_time ≤ its own signal_time — no lookahead):
  - **symbol gate**: suppress a new long on a name if that name had a failure
    confirmation in the prior 32 bars (8h).
  - **market gate**: suppress new longs market-wide when ≥3 distinct names failed
    in the prior 16 bars (4h) — the doc's S6 breadth-collapse.
- Gated longs free basket capacity that first-come selection backfills, so the
  comparison answers: did standing aside beat carrying these longs?

## Evidence

### Symbol risk-off improves the book at every capacity

net20 / max-drawdown-proxy / return-per-drawdown:

| max | baseline | symbol_risk_off | market_risk_off |
|---|---|---|---|
| 5 | 10.62% / -6.13% / 1.73 | **13.27% / -4.38% / 3.03** | 8.53% / -4.29% / 1.99 |
| 8 | 10.91% / -4.08% / 2.67 | **11.74% / -2.88% / 4.08** | 7.97% / -3.03% / 2.63 |
| 10 | 9.60% / -3.41% / 2.82 | **10.26% / -2.45% / 4.19** | 7.13% / -2.57% / 2.78 |

The symbol gate raises net, cuts the drawdown proxy ~30%, and lifts
return/drawdown 50–75% at all three capacities — and the supporting guards move
the right way too: month-cap net rises (max8: 0.53%→0.72%) and max monthly
contribution falls (max8: 0.42→0.34, i.e. less concentration). Consistency across
max5/8/10 makes this a structural effect, not a single-trade fluke.

The market-breadth gate does the opposite to return: it removes 59 longs
market-wide and net falls 3pp, because breadth-elevated windows overlap with the
co-impulse environments where the CIC longs actually work. It buys some drawdown
but pays too much net. Combined ≈ market (market dominates). **Only the
symbol-level gate is worth carrying forward.**

### Honest mechanism: this is risk reduction, not loser-removal

The 34 symbol-gated longs were on average mildly *positive* (gated_realized_net
+0.48%), so the gate is **not** simply deleting losers. The improvement is a
risk-adjusted reshuffle: standing aside on a name that just failed a reclaim
avoids re-entering a churning/toppy name, and the freed capacity backfills with a
fresher setup (the v1.1 result that skipped ≈ selected makes the backfill roughly
free). Net comes out flat-to-up while drawdown and concentration drop materially.
The honest claim is therefore **better risk-adjusted return**, headlined by the
drawdown and concentration cuts, with net a modest bonus — not a net-alpha story.

## Interpretation

1. The short-side research delivers value exactly where the doc predicted — as a
   long risk-off gate, not a short book. v1.2s said "don't short these"; v1.2s2
   shows "but do stand aside on a name that just failed, and the long book gets
   safer."
2. The *symbol-local* signal is the useful one. A failed reclaim / crowded unwind
   on a specific name is a reason to skip new longs on *that* name. The
   *market-wide* breadth read is too coarse here and throws away good co-impulse
   longs — consistent with v1.2s S3/S5 only working in BTC_down, not as a blanket
   market timer.
3. This is the cleanest near-term integration of the short work into the live
   product: it needs no short execution, no borrow, no squeeze risk — it only ever
   *removes* longs.

## Next steps (phase 3, not done here)

- Port the symbol gate into the live long pipeline as a per-symbol "no new long
  for N bars after a failure confirmation" rule and shadow it against the live
  CIC book (still research/shadow — no real-live).
- Tune the cooldown window (32 bars was a first guess; the drawdown gain may peak
  elsewhere) and test a *size-down* variant (half size instead of full skip).
- Revisit the market gate only with a much higher breadth threshold or BTC_down
  conditioning, so it stops vetoing healthy co-impulse windows.

## Phase-3: tuning, size-down, and a live shadow gate

### Cooldown sweep (symbol gate, max8, full-skip)

| cooldown | net20 | max_dd | ret/dd | gated |
|---|---|---|---|---|
| 8 (2h) | 11.73% | -3.66% | 3.21 | 4 |
| 24 (6h) | 12.06% | -2.88% | 4.19 | 33 |
| 32 (8h) | 11.74% | -2.88% | 4.08 | 34 |
| **48 (12h)** | **12.24%** | **-2.50%** | **4.90** | 40 |
| 64 (16h) | 11.62% | -2.50% | 4.65 | 43 |
| 96 (24h) | 11.56% | -2.50% | 4.63 | 44 |

Drawdown improves monotonically with the cooldown and plateaus around 48–96 bars;
net peaks at **48 bars (12h)**. 48 is the tuned default: net 12.24%, drawdown
-2.50% (vs baseline -4.08%, a 39% cut), ret/dd 4.90 (vs 2.67).

### Full-skip beats size-down

A half-size variant (keep the gated long at 0.5× P&L instead of skipping it)
helps drawdown but trails full-skip on every axis (cd=48: net 10.46% / dd -3.18%
/ ret-dd 3.29 vs full-skip 12.24% / -2.50% / 4.90). **Standing fully aside beats
sizing down** — consistent with the mechanism (the freed slot backfills with a
fresher setup, which a half-size long blocks).

### Live shadow gate (behavior-preserving)

`pressure_graph.live.risk_off_gate` packages the gate for the live pipeline:
`build_risk_off_events`, `annotate_signals_with_risk_off`, and a single-point
`risk_off_decision(symbol, time, events)` API. It is wired into the v07d2
paper-live run-once script as **shadow only** (guarded, additive): each refresh
writes `risk_off_shadow/risk_off_signal_shadow.csv` recording which live CIC long
signals *would* be suppressed, accruing the gate's would-be decisions for future
paper-live validation. No paper trade is changed.

## Reproduction

```
pressure-graph run-v12s2-risk-off --config configs/v0_3.yaml   # research overlay + sweep
python scripts/v07d2_live_once.py                              # writes risk_off shadow CSV
```

Requires the v0.3 feature parquet and the v0.9D capacity trade cache. Outputs in
`reports/v1_2s2_long_risk_off_overlay/`: overlay_summary, cooldown_sweep,
risk_off_events, breadth_timeline, candidate_notes.md. 120 tests green on the box.

No paper-live / real-live permission changes — the live wiring is shadow-only.

