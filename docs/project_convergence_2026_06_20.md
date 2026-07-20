# Project Convergence - 2026-06-20

This document consolidates the long-side stack, the pulled short-side research,
and the current diagnostic lines into one operating state. It does not change
any live permission by itself.

## Executive State

The project is now in convergence mode, not broad discovery mode.

Current deployable-looking object:

- Long core: `P2 CIC1+CIC2 max8`
- Long management: `O6 late-burst overflow`, `CP60`, `Protect_A cap2`
- Risk context: failure/risk-off overlay, low-coimpulse router diagnostic,
  token/on-chain attention diagnostic

Current permission:

- Paper/shadow logging: allowed
- Real-live: disabled
- Canary-live: disabled
- New broad historical rule mining: paused
- Diagnostic data collection: allowed

The correct next work is forward evaluation, sample sufficiency, execution
realism, and demotion/promotion discipline.

## Active Benchmark Stack

Keep the active benchmark set small.

| ID | Structure | Status | Use |
|---|---|---|---|
| S0 | `P2 max8` | core forward baseline | compare all stack changes against it |
| S3 | `P2 max8 + CP60 + O6` | conservative managed shadow | current conservative managed long stack |
| S5 | `P2 max8 + Protect_A cap2 + O6` | research-improved shadow | test whether Protect_A can replace CP60_all behavior |

Interpretation:

- `P2 max8` is the core signal pool and capacity expression.
- `O6` is an additive overflow sleeve, not a selector.
- `CP60` is weak-position pruning, not entry alpha.
- `Protect_A cap2` is a CP60 false-exit protection candidate.
- None of these are real-live ready.

## Short-Side Convergence

The pulled short-side work reinforces the same conclusion: failure information
is useful as long-book risk context, not as an automated short engine.

Closed as automated short alpha:

- Standalone short motifs
- S1/S3/S5 failed-reclaim breakdown shorts
- CIC-failure shorts
- Crowded-stall automated short sleeve
- Relative-value beta short pairs
- A1 cross-exchange downside lead-lag short as a strategy

Retained:

- `F3` no-overflow risk layer: shadow/counterfactual only
- `F5` CIC2-only same-symbol no-long risk layer: shadow/counterfactual only
- A1/downside lead-lag: regime-event diagnostic only
- Crowded-stall/funding-crowding: diagnostic only

Hard rule:

No short candidate may be re-opened unless it satisfies the short reopen
criteria before the production run:

1. At least three-month distributed sample with no single month contributing
   more than 35% of alpha.
2. Hedge sign correct: correlation versus long-stack monthly net <= -0.3, or
   positive PnL in the long stack's worst month.
3. Survives 30bp cost plus 5bp slippage.
4. Strictly beats `no_long`.
5. Short squeeze margin <= 20%.

## Sell-Pressure Propagation Map

The sell-pressure propagation work is allowed as a map, not a strategy.

Allowed:

- Build edge maps.
- Report shuffled-null and bootstrap confidence.
- Identify source-target lag relationships.
- Record diagnostic state for future research.

Not allowed:

- Auto-short from propagation edges.
- Convert S1/S3/S5 or CIC-failure into shorts.
- Promote a propagation edge without the global validation gates below.

Minimum map edge criteria before any further action:

- `n_events >= 30`
- active months >= 3
- max month share <= 50% for a map edge, <= 35% for a strategy candidate
- bootstrap CI low > 0 versus shuffled-null
- adverse upsample <= 40%

## Diagnostic Lines Kept

These stay in the forward ledger, but cannot act on trades yet:

- Low-coimpulse/router risk score
- Failure/risk-off overlay
- Token/on-chain attention
- Perp crowding/funding/OI state
- Cross-exchange source context
- Orderflow/orderbook coverage
- Narrative/sector context
- Listing/catalyst atlas

Their purpose is explanation and future attribution, not selection, sizing,
gating, or live permission.

## Rejected Or Paused Lines

Do not keep trying to rescue these without a genuinely new data source or a
pre-registered reopen condition:

- MIR1 raw as primary
- Directed leader-beta causality
- LBR/LBC paper-live
- Cluster impulse as a primary gate
- Static orderbook ask-thin/upside-vacuum ranking
- 15m aggregate orderflow ranking
- Generic top-k CIC ranking
- Fixed burst budget, delay allocation, simple replacement
- Pure 2h time stop
- Standalone short and pair-RV short
- A7 cross-exchange lag pocket as live/shadow selector
- Funding/OI standalone long or short
- Real sector labels as a selector
- Listing 24h chase

## Global Validation Gate

Every future candidate, long or short, must report the same core checks:

1. Sample size by action/component.
2. `net20`, `net30`, and cost stress through at least 30bp.
3. Best-month share <= 35% for any promotable candidate.
4. `month_cap35_net20 > 0`.
5. Leave-one-month does not collapse.
6. Max symbol contribution <= 35%.
7. Disjoint walk-forward buckets do not include a clearly negative bucket.
8. Bootstrap CI on mean net does not straddle zero for promote-tier claims.
9. Candidate beats matched random p75, preferably p90.
10. Domain-specific controls pass, e.g. no-long baseline for shorts,
    selected/skipped for capacity rules, shuffled graph controls for graph
    edges, and same-token/random-token controls for token attention.

If a candidate fails concentration or walk-forward, it cannot be promoted even
when headline net is strong.

## Forward Sample Thresholds

Do not evaluate short windows before these are met:

| Component | Initial evaluation threshold | Stronger review threshold |
|---|---:|---:|
| Core P2 trades | 100 | 200 |
| CP60 exits | 50 | 100 |
| Protect_A protected exits | 30 | 50 |
| O6 overflow trades | 30 | 50 |
| Failure risk-off suppressions | 50 | 100 |
| Token-prior P2/CIC trades | 100 | 200 |

Before a threshold is met, use:

- `sample_status = insufficient`
- `evaluation_status = no_decision`

## Promotion / Demotion Rules

Promotion from shadow to stronger paper candidate requires:

- Forward sample threshold is met.
- `net20` beats the relevant baseline.
- `net30` remains positive.
- `month_cap35_net20` remains positive.
- Worst month and worst burst do not worsen versus baseline.
- Cost stress at 30bp does not collapse the edge.
- Execution realism and risk envelope checks pass.

Demotion should happen when:

- Core trades >= 100 and core `net20 <= 0`.
- O6 overflow trades >= 30 and incremental O6 contribution <= 0.
- CP60 exits >= 50 and CP60 delta <= 0.
- Protect_A protected exits >= 30 and protect delta <= 0.
- Failure risk-off suppressions >= 50 and suppression harms net and drawdown.

## Operating Loop

1. Keep S0/S3/S5 forward logging running.
2. Keep F3/F5, low-coimpulse, token attention, orderflow/orderbook, and
   sell-pressure propagation as diagnostics.
3. Run weekly promotion audit only when fresh paper-live data exists.
4. Re-run `v2.3 -> v2.4 -> v2.5 -> v2.6` for the long stack after each
   meaningful sample refresh.
5. Re-run failure/risk-off audit only as a long-side overlay, not a short sleeve.
6. Do not start a new broad historical alpha line unless it is backed by a new
   information source and a pre-registered validation plan.

## Current Decision

No promotion. No real-live. No canary-live.

The project is not dead; it is at the point where discipline matters more than
cleverness. The live system should now answer one question with future data:

Can `P2 max8`, with O6/CP60/Protect_A/failure-risk diagnostics, remain positive
and controlled under forward sample and real execution assumptions?
