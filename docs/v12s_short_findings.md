# v1.2s Short Motif Atlas — Findings (2026-06-12)

Phase-1 response to the short instruction doc. Tier: **research only** — nothing
here is wired to shadow, paper-live, or real-live.

## Verdict (one line)

**None of the four failure motifs is a standalone short. Every one is net-negative
across all cost levels, and for every motif the value of the signal is captured by
*not being long* — opening a short on top of that loses money.** This is exactly
the outcome the doc warns to expect, now backed by a year of Top30 data.

## What was tested

The doc's stance, implemented literally: shorts are strong→weak *failure unwinds*,
not inverted longs, and the dominant risk is the squeeze. Four per-symbol failure
motifs over the 78-symbol Top30 union (1y, 15m), each entered short at the next bar
open after a failure *confirmation*:

| code | motif | trigger → confirmation |
|---|---|---|
| S1 | failed_reclaim_short | bullish shock → pullback → reclaim attempt → **failed** reclaim |
| S2 | extreme_exhaustion_short | extreme strength + upper-wick rejection → breakdown |
| S3 | crowded_long_unwind_short | high funding + high OI Δ + stalled price → support break |
| S5 | btc_down_breakdown_short | BTC_down + symbol breakdown → failed bounce → lower low |

S4 (leader→beta failure) was **deferred** rather than faked: it needs the
cross-sectional graph, and a per-symbol proxy would misrepresent a graph claim —
which the doc explicitly cautions against.

Risk accounting follows doc §5: Top30 liquidity only, a shorter validity window
(max-hold 16 bars), an asymmetric squeeze-averse exit (cover 3% lower / stop 2.5%
higher, stop-first on ambiguous bars), and a **+5bp short slippage add-on** on top
of the 10/20/30/50bp cost grid.

Each motif is scored against three count-matched, deterministically-seeded
baselines — entry-only (trigger without the failure confirmation), matched-random
(same-symbol random bars), and plain-drop (普通下跌后做空, short after a naive ~1h
drop) — plus a BTC-regime split, month-cap, and symbol-contribution guards. 5,236
real short trades in total.

## Evidence

### 1. No standalone short edge (20bp + 5bp short slippage)

| motif | trades | short_net | win | hit_down_3pct | squeeze_out | avg MAE |
|---|---|---|---|---|---|---|
| S1 failed_reclaim | 316 | **-0.331%** | 46.5% | 28.5% | 33.2% | 1.73% |
| S2 extreme_exhaustion | 854 | **-0.556%** | 39.5% | 18.9% | 29.9% | 1.64% |
| S3 crowded_long_unwind | 999 | **-0.422%** | 36.0% | 10.8% | 12.1% | 1.14% |
| S5 btc_down_breakdown | 3067 | **-0.329%** | 39.1% | 19.0% | 17.9% | 1.41% |

Every motif loses per trade. Cost stress only deepens it (S5 short_net: -0.13% at
10bp → -0.93% at 50bp); no motif crosses zero at any cost. Month-cap haircuts are
~0 because no motif has a positive total to cap.

### 2. The failure structure adds information — but not enough to trade

Lift of the real motif over its baselines (net20):

| motif | vs entry_only | vs matched_random | vs plain_drop |
|---|---|---|---|
| S1 | **+0.127%** | **+0.117%** | +0.007% |
| S2 | -0.093% | -0.110% | -0.121% |
| S3 | +0.017% | +0.019% | +0.039% |
| S5 | -0.079% | +0.098% | **+0.106%** |

S1's failed-reclaim confirmation genuinely beats both naive shorting and its own
entry-only trigger — the failure structure *carries information*. But the absolute
level is still negative: the edge is real and too small to clear cost + squeeze.
S2 is actively worse than its baselines (the upper-wick exhaustion proxy is
mistimed); S3/S5 only marginally beat plain-drop.

### 3. The decisive test — short vs just-not-being-long (doc §8)

For every motif, the forward long loss avoided by being flat dwarfs the short P&L:

| motif | short_net | avoided_long_loss | best use |
|---|---|---|---|
| S1 | -0.331% | **+1.027%** | long risk-off |
| S2 | -0.556% | **+0.833%** | long risk-off |
| S3 | -0.422% | **+0.677%** | long risk-off |
| S5 | -0.329% | **+0.823%** | long risk-off |

This is the doc's central claim, confirmed without exception: **these signals'
value is to cut long exposure / forbid new longs, not to open shorts.** A book that
simply stands aside at these moments captures 0.7–1.0% of avoided drawdown per
event; a book that shorts hands most of that back via cost and squeeze.

### 4. Squeeze sanity (regime split)

net20 / squeeze-out rate / n, by BTC regime:

| motif | BTC_down | BTC_chop | BTC_up |
|---|---|---|---|
| S1 | -1.94% / 65% / 17 | -0.07% / 29% / 226 | -0.78% / 40% / 73 |
| S2 | -1.70% / 67% / 24 | -0.51% / 30% / 600 | -0.57% / 26% / 230 |
| S3 | **+0.22% / 18% / 78** | -0.48% / 12% / 919 | -1.51% / 50% / 2 |
| S5 | -0.25% / 17% / 2317 | -0.57% / 21% / 749 | -3.00% / 100% / 1 |

Two things the doc predicted show up cleanly:
- **BTC_up shorts get squeezed.** S3/S5 in BTC_up squeeze 50–100%; the few trades
  there are the worst cells in the table. Never short strength into a rising BTC.
- **Shorting *into* a falling BTC also squeezes violently.** S1/S2 in BTC_down
  squeeze 65–67% — counterintuitive but real: chasing a breakdown after BTC is
  already down catches the snapback, exactly the "下跌中继 / 迅速反抽" trap from doc
  §2.5 / §2.E.

The single mildly-positive cell is **S3 (crowded long unwind) in BTC_down**:
+0.22% over 78 trades with the lowest squeeze rate (18%). It is not significant on
its own, but it is the only place the data hints that crowded-unwind shorts could
have an edge — and notably it is *not* a "price fell so short it" cell.

## Interpretation

1. The doc's guardrails were right. Inverting-long instincts (S2 exhaustion,
   S5 breakdown-continuation) are the worst performers and the most squeeze-prone;
   the more "failure-structured" S1 and the crowded-unwind S3 carry the only real
   information, and even they don't clear costs as shorts.
2. The first deliverable is therefore not a short strategy but a **risk-off
   signal**: route these confirmations into the long book as "reduce / forbid new
   longs", which is where 0.7–1.0% per event actually lives.
3. If a standalone short is pursued later, the only data-supported lead is
   **S3 crowded-long-unwind, gated to BTC_down**, with tighter squeeze control —
   not breakdown-continuation, and not exhaustion-shorting.

## Next steps (phase 2, not done here)

- Wire S1/S3/S5 confirmations as a **long-risk-off gate** and measure the long
  book's drawdown reduction (the actual product of this research).
- S4 leader→beta failure as a real cross-sectional pass on the graph, once the
  long-side graph edges are trusted.
- For S3-in-BTC_down only: orderflow-confirmed exhaustion (taker sell pressure,
  failed-bid-rebuild) per the long-side v1.1 tooling, to see if the squeeze rate
  can be driven below cost.

## Reproduction

```
pressure-graph run-v12s-short-atlas --config configs/v0_3.yaml
```

Requires the v0.3 feature parquet (built by the v11 chain). Outputs in
`reports/v1_2s_short_motif_atlas/`: candidate_summary, baseline_comparison,
regime_split, month_cap, symbol_contribution (the doc's mandated files) plus
short_motif_trades and candidate_notes.md. 109 tests green on the clean box.

No shadow / paper-live / real-live permission changes.

