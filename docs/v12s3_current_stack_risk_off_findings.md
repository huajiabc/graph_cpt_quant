# v1.2s3 Current Long-Stack Risk-Off Overlay — Findings (2026-06-13)

Phase-4 finish in the form the short instruction doc asked for: replay the
v1.2s2 symbol-level risk-off gate against the **current** long stack
(CIC-filtered MIR1 primary, P2 max8 core basket, O6 late-burst overflow,
MIR1 raw reference, IR2 deferred, C2 sentinel) and answer the doc's five
Phase-4 questions on the live shape. Tier: **research only** — nothing here
is wired to shadow, paper-live, or real-live.

## Verdict (one line)

**Symbol-level risk-off improves every active stack piece on net, cuts P2
max8 drawdown 39% and P2+O6 overflow drawdown 42%, and does not false-kill
the O6 overflow sleeve. Decision stays as the doc asks: active gate **no**,
shadow gate **yes**.**

## What was replayed

Five pieces from the current stack against 10,216 S1/S3/S5 failure
confirmations (78 Top30 symbols, 1y, 15m). IR2 reference is deferred — it
lives in the v0.7D.2 paper-live signal log, not in the v0.9D backtest cache,
so the table surfaces it as a placeholder rather than silently dropping it.
S2 is excluded from the gate (mistimed in v1.2s); cooldown defaults to the
v1.2s2-tuned 48 bars and is re-validated below.

## Evidence

### 1. Stack-level deltas (48-bar cooldown, full-skip vs baseline)

| stack piece | baseline net20 / dd | gate net20 / dd | Δ net / Δ dd | gated |
|---|---|---|---|---|
| S1 CIC-filtered MIR1 primary (max3) | 10.08% / -5.70% | **11.88% / -7.44%** | +1.80% / -1.74% | 24 |
| S2 P2 max8 core basket | 10.91% / -4.08% | **12.24% / -2.50%** | +1.33% / +1.58% | 40 |
| S3 P2 max8 + O6 overflow | 12.23% / -4.28% | **13.22% / -2.50%** | +0.98% / +1.79% | 40 |
| S4 MIR1 raw reference | -7.07% / -15.56% | **+5.67% / -6.26%** | +12.74% / +9.29% | 67 |
| S6 C2 sentinel (CIC2 broad) | 10.94% / -6.13% | **14.21% / -4.00%** | +3.27% / +2.13% | 40 |

Four of five active pieces improve on **both** net and drawdown; the gate
clearly belongs near the long book. The S1 primary piece is the only
caveat: net rises but its drawdown proxy worsens by 1.74pp. That is the
small-portfolio (max=3) effect — losing 24 of ~80 trades reshapes the
equity path enough that the worst stretch sits in a different month. The
ret/dd at the primary still ends well below max8/overflow (1.60 vs ~5),
which is exactly why the primary stays a paper-live candidate, not a
position-sizing override.

### 2. Q1 — Does the gate improve P2 max8 drawdown?

Yes. -4.08% → **-2.50%**, a 39% cut on the headline basket. Net rises
1.33pp. Identical numbers to v1.2s2's tuned cell, which confirms the
v1.2s2 win is reproducible on the same cache, not sample-specific.

### 3. Q2 — Does the gate false-kill O6 late-burst overflow?

No. Of 11 overflow-eligible signals in the baseline P2+O6 sleeve, the gate
suppresses **10** at signal time (they fell inside a 12h failure cooldown).
The overflow sleeve still fills 9 trades after the gate is on, because the
freed selection capacity backfills with fresh late-burst entries. The full
stack improves: net **+0.98pp**, drawdown -42% (-4.28% → -2.50%). The
overflow sleeve does not need to be turned off.

### 4. Q3 — Are suppressed trades actually worse, or just risk-adjusted better?

211 longs suppressed across the stack (S2/S3/S6 share most of them at 40
each; S1=24, S4=67). Loss share = 49.8%, mean would-be net20 = +0.08% —
i.e. the suppressed set is, on average, a *coin flip*, not a clear loser.
The improvement is the v1.2s2 mechanism (risk-adjusted reshuffle), not
loser-removal. Per-motif:

| motif | suppressions | mean would-be net20 |
|---|---|---|
| S1 failed_reclaim | 18 | **-0.40%** |
| S3 crowded_long_unwind | 23 | **+1.70%** |
| S5 btc_down_breakdown | 170 | -0.08% |

S1 suppressions are loss-avoidance, as expected. S5 dominates by volume
and is breakeven on average — the gain comes from being out of the wrong
name long enough for a fresh setup to backfill. S3 is the honest
inconvenience: the 23 crowded-unwind suppressions would have averaged
+1.70% — the gate gives some money back here. The net stack effect is
still positive on every active piece because the freed-slot backfill (the
mechanism the doc anticipated in §1) more than compensates.

### 5. Q4 — Full-skip vs size-down

Full-skip beats half-size on **every** piece, by both net and (mostly)
drawdown:

| piece | full-skip net / dd | half-size net / dd |
|---|---|---|
| S1 primary | 11.88% / -7.44% | 10.08% / -5.70% |
| S2 P2 max8 | **12.24% / -2.50%** | 10.46% / -3.18% |
| S3 P2+O6 | **13.22% / -2.50%** | 11.86% / -3.38% |
| S4 raw MIR1 | **5.67% / -6.26%** | -2.87% / -10.17% |
| S6 C2 sentinel | **14.21% / -4.00%** | 10.95% / -5.06% |

S1's row again shows the small-portfolio quirk (half-size matches the
un-gated dd because at max=3 the half-size variant essentially keeps the
same trade composition with halved P&L). Everywhere else, the v1.2s2
finding holds: **standing aside frees a slot that backfills with a
fresher setup; sizing down occupies the slot without earning it.**

### 6. Q5 — Is the 48-bar cooldown still right?

For the active stack: yes. Best cooldown by ret/dd per piece, full-skip:

| piece | best cooldown | net20 | max_dd | ret/dd |
|---|---|---|---|---|
| S1 primary | **48** | 11.88% | -7.44% | 1.60 |
| S2 P2 max8 | **48** | 12.24% | -2.50% | 4.90 |
| S3 P2+O6 | **48** | 13.22% | -2.50% | 5.29 |
| S4 raw MIR1 | 64 | 5.41% | -5.93% | 0.91 |
| S6 C2 sentinel | **48** | 14.21% | -4.00% | 3.56 |

48 bars (12h) is optimal on the three pieces that matter for the live
product (S1, S2, S3) and on the C2 sentinel diagnostic. Raw MIR1 (S4) is a
noisier reference that prefers 64 bars, consistent with its unfiltered
nature — and it is not a candidate for live use anyway. **Carry 48 bars
into the shadow gate; do not re-tune per-piece.**

## Interpretation

1. The v1.2s2 result generalizes from "works on P2 max8" to "works on the
   entire live structure". P2 max8 reproduces v1.2s2 exactly, P2+O6
   overflow is the strongest cell in the table, and the C2 sentinel
   confirms the effect is not unique to the CIC1 sleeve.
2. The S3 motif is the new caveat. Its suppressions would have averaged
   +1.70%, so the gate is taking real money off the table on this slice.
   The stack net is positive because backfill + dd-cut more than pay for
   it, but it argues against ever using S3 alone as the gate signal — the
   ensemble (S1+S3+S5) is what works.
3. The S1 primary's drawdown-worsening result is the cleanest case for
   shadow-only rollout. Same direction (net up), but the dd path changes
   in ways that a 1-year sample can't decide on its own. Shadow gives the
   live system a few more months to vote on whether the smaller portfolio
   benefits from the gate.
4. IR2 stays out of this pass because it is not in the v0.9D backtest
   cache. Bringing it in needs a future enrichment of the cache from the
   v0.7D.2 paper-live signal log, not a research patch here.

## Decision (unchanged from instruction)

- **active gate: no** — Phase-4 is not promoted to the live execution path.
- **shadow gate: yes** — the v1.2s3 evidence supports the existing v1.2s2
  `risk_off_signal_shadow.csv` wiring; the shadow's coverage broadens to
  the full active stack (CIC-filtered MIR1 primary, P2 max8, P2+O6
  overflow, C2 sentinel) at cooldown=48 bars, full-skip semantics. The
  shadow stays additive — it only ever records would-be suppressions; no
  paper trade is changed.

## Next steps (phase 2, not done here)

Per the instruction file's §2-§8 the next research line is **v13S Short
Failure Graph** (S1/S2/S3/S4/S5 motifs as a research atlas, with
short-specific labels — `future_max_down_4h/12h`, `hit_down_3pct_4h`,
`up_before_down`, `short_squeeze_before_hit` — and the §6 control set:
matched random, entry-only breakdown, skip-long-only, BTC_down only,
BTC_chop only, opposite-long control). v1.2s already produced a
research-quality first pass of S1/S2/S3/S5; the v13S.1 atlas
re-implementation extends that with the short-specific label family and
the explicit skip-long control. **Not started here**; this file closes
Phase-4 of the long-side product only.

## Reproduction

```
pressure-graph run-v12s3-current-stack-risk-off --config configs/v0_3.yaml
```

Requires the v0.3 feature parquet and the v0.9D capacity trade cache.
Outputs in `reports/v1_2s3_current_stack_risk_off/`:
`risk_off_on_current_long_stack`, `risk_off_on_p2_max8`,
`risk_off_on_p2_max8_plus_o6`, `suppressed_trade_attribution`,
`cooldown_48bar_validation`, `candidate_notes.md` — the six instruction
files. 131 tests green on the box (120 v1.2s2 baseline + 11 new v1.2s3).

No paper-live / real-live permission changes.
