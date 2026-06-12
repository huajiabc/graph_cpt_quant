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

## Reproduction

```
pressure-graph run-v12s2-risk-off --config configs/v0_3.yaml
```

Requires the v0.3 feature parquet and the v0.9D capacity trade cache. Outputs in
`reports/v1_2s2_long_risk_off_overlay/`: overlay_summary, risk_off_events,
breadth_timeline, candidate_notes.md. 113 tests green on the clean box.

No paper-live / real-live permission changes.

