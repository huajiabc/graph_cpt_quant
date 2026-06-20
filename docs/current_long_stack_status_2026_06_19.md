# Current Long Stack Status - 2026-06-19

This document is a navigation snapshot for the current long-stack research state. It does not change any live permission.

## Permissions

- Paper/shadow logging: allowed
- Real-live: disabled
- Canary-live: disabled
- New broad historical rule mining: paused
- Diagnostic data collection: allowed

## Active Long Stack

Core:
- P2 CIC1+CIC2 max8

Management layers:
- O6 late-burst additive overflow
- CP60 weak-position pruning
- Protect_A beta-high cap2 false-exit protection

Diagnostics:
- Low-coimpulse / router risk state
- Token-level DEX attention context
- Failure/risk-off overlay reference

## Current Benchmarks

- S0: P2 max8 baseline
- S3: P2 max8 + CP60 + O6
- S5: P2 max8 + Protect_A cap2 + O6

Current decision:
- S3: keep conservative shadow, needs more sample
- S5: keep research-improved shadow, needs Protect_A / O6 forward sample
- Real-live remains disabled

## Forward Sample State

Latest v2.4 sample gaps:
- S0 P2 max8: core=12, CP=0, protected=0, overflow=0
- S3 P2 max8 + CP60 + O6: core=12, CP=7, protected=0, overflow=0
- S5 P2 max8 + Protect_A cap2 + O6: core=12, CP=7, protected=0, overflow=0

Interpretation:
- CP60 has live behavior evidence but insufficient evaluation sample.
- O6 has no current forward trigger sample.
- Protect_A cap2 has no current protected-exit forward sample.

## Execution Realism

Latest v2.5 decision:
- Not execution-realism ready.
- No structure passes current cost stress.
- Real-live blockers: cost ladder partial, CP60 sample too small, O6 sample missing, Protect_A sample missing, funding/contract filters not finalized, native depth/slippage not finalized.

## Risk Envelope

Draft envelope:
- core_max_positions = 8
- overflow_max_slots = 4
- overflow_total_exposure_cap = 2 units
- total_exposure_cap = 10 units
- daily_new_exposure_cap = 8 units proposed
- rolling_4h_new_exposure_cap = 6 units proposed

Latest v2.6 decision:
- checks_passed = 1/6
- canary_or_real_live_ready = false
- Envelope is documented, not finalized for live promotion.

## Token Attention

v6.7 status:
- Forward/counterfactual logging only.
- No selector, gate, sizing, shadow portfolio, or real-live permission.

Main token-prior 24h decisions:
- CIC1: forward counterfactual candidate
- P2_all: diagnostic context
- O6_late9: diagnostic context
- CIC2: diagnostic only

Current paper-live output now includes:
- reports/v0_7d2_cic_mir1_paper_live/token_attention_counterfactual_live.csv
- reports/v0_7d2_cic_mir1_paper_live/token_attention_counterfactual_live.parquet

## Diagnostic / Rejected Lines

Diagnostic only:
- low-coimpulse router
- failure risk-off overlay
- token/on-chain attention context
- cross-exchange v4
- perp crowding v5
- narrative sector v7
- listing/catalyst v8

Rejected or paused as deployable selectors:
- 15m orderflow ranking
- static orderbook ranking
- standalone short
- directed leader-beta
- flush proxy
- generic short motif

## Primary Report Entrypoints

- reports/v2_3_forward_evaluation_decision_ledger/candidate_notes.md
- reports/v2_4_long_stack_promotion_audit/promotion_decision.md
- reports/v2_5_execution_realism_audit/candidate_notes.md
- reports/v2_6_risk_envelope_finalization/candidate_notes.md
- reports/v6_7_token_attention_forward_context/candidate_notes.md

## Next Operating Loop

1. Keep P2/S3/S5 forward logging running.
2. Do not evaluate short windows before sample thresholds.
3. Re-run v2.3 -> v2.4 -> v2.5 -> v2.6 after each paper-live refresh.
4. Watch sample thresholds:
   - core trades >= 100
   - CP60 exits >= 50
   - protected exits >= 30
   - overflow trades >= 30
5. Keep token attention as counterfactual context until future sample confirms it.
