# Research Consolidation And Paper-Live Status - 2026-06-19

This note consolidates the current research state after the v0.7-v6.5 experiments, and records the latest local paper-live snapshot after a manual refresh on 2026-06-19 Asia/Shanghai time.

## Executive Decision

The project should pause broad rule discovery for now.

The strongest current structure is still the long-side CIC/P2 stack:

- Core signal pool: P2 CIC1 + CIC2 combined, max8 basket.
- Overflow sleeve: O6 late-burst additive overflow, shadow only.
- Post-entry management: CP60 weak-position pruning, shadow only.
- False-exit protection: Protect_A beta-high cap2, research-improved shadow candidate.
- Risk diagnostics: low-coimpulse/router score, failure risk-off overlay, on-chain attention, crowding, orderflow/orderbook diagnostics.

Real-live remains disabled. The correct next operating mode is forward evaluation and promotion audit, not more historical rule mining.

## Latest Paper-Live Snapshot

Local paper-live was stale before refresh:

- Before manual refresh: latest processed feature time was 2026-06-15 12:30 UTC.
- After manual refresh: latest feature time is 2026-06-18 18:15 UTC.
- API probe: OK.
- Local processed data: fresh.
- Processed symbols: 50.
- Processed rows: 169,044.

Current market state:

- BTC state: BTC_down.
- volume_impulse_density: 0.0.
- market_volume_impulse_density_high: false.

Current v0.7D.2 primary snapshot:

- Total candidate signals: 639.
- Primary portfolio trades: 1.
- Primary trade: FARTCOINUSDT, CIC1_FILTERED_MIR1.
- Entry: 2026-06-15 14:15 UTC.
- Exit: 2026-06-15 17:30 UTC.
- Exit reason: SL.
- Primary average net10: -3.2000%.
- Primary average net20: -3.4000%.
- Sample status: insufficient.
- Evaluation status: no_decision.

This 1-trade result should not be used to evaluate the strategy.

## Current Live/Shadow Modules

| Module | Current Status | Latest Observation | Decision |
|---|---:|---|---|
| P2 max8 core | Shadow / paper candidate | 1 selected trade, net20 -3.4000% | Need more sample |
| O6 late-burst overflow | Shadow only | 0 overflow triggers in current window | No decision |
| CP60 | Shadow only | 1 checkpoint exit reduced the FARTCOIN loss | Behavior check passed, sample insufficient |
| Protect_A cap2 | Research-improved shadow candidate | 0 protected exits in current window | No decision |
| Failure risk-off | Diagnostic/shadow overlay | 263 failure events; 20/192 long signals would be suppressed | Keep diagnostic |
| Short S2 shadow | Research only | 8 trades, negative net20 | Do not upgrade |
| Orderflow shadow | Data layer | Cache refreshed, but reclaim-window coverage remains 0 for the live trade set | Data layer only |

CP60 latest behavior:

- Baseline selected net20: -3.4000%.
- CP60 selected net20: -1.7355%.
- Baseline portfolio net20: -0.4250%.
- CP60 portfolio net20: -0.2169%.
- New trades due to released slot: 0.

Interpretation: CP60 again behaved as weak-position pruning, not slot-capture alpha. This matches the offline thesis, but the live sample is still far too small.

## Research Line Decisions

### Keep As Active Forward Stack

P2 CIC1+CIC2 max8:

- Best current long basket structure.
- Not a top-k selector.
- Needs forward sample before promotion.

CP60:

- Valid weak-position pruning candidate.
- Needs at least 50 checkpoint exits for initial evaluation.

O6:

- Late-burst additive overflow remains plausible.
- Needs at least 30 overflow trades before initial evaluation.

Protect_A cap2:

- Offline stability passed leave-one-trade, leave-one-burst, leave-one-month, and concentration cap checks.
- Needs forward protected exits before replacing CP60_all.

### Keep As Diagnostics

Low-coimpulse / router diagnostics:

- Useful explanation of weak holdout regimes.
- Not stable enough as live action.

Failure risk-off:

- Standalone short failed.
- Symbol-level no-long overlay is the useful form.
- Keep as long risk-layer diagnostic/shadow.

On-chain attention:

- Market-level propagation diagnostic passed.
- Token-level coverage improved but token-prior P2/CIC/O6 samples remain below attribution thresholds.
- Keep in forward ledger.

Cross-exchange v4:

- A7 lag pocket is right-tail/regime diagnostic only.
- Binance taker-buy is diagnostic context only.

Perp crowding v5:

- Funding/OI states are diagnostic only.
- Funding extreme + OI low RV failed strict pair-cost test.

Orderbook/orderflow:

- Static orderbook ranking failed.
- 15m aggregate orderflow ranking failed as selector.
- Keep data collection for future diagnostics only.

Narrative/listing/catalyst:

- Real sector labels did not beat random sector.
- Listing 24h chase is bad; 7d digestion is only an atlas clue.
- No deployable catalyst alpha yet.

### Rejected Or Paused

- MIR1 raw as primary.
- Leader-beta directed causality.
- LBR/LBC paper-live.
- Cluster impulse as primary gate.
- Static orderbook ask-thin/upside-vacuum selector.
- 15m orderflow capacity ranking.
- Top-k ranking as the main CIC capacity solution.
- Fixed burst budget / delay allocation / simple replacement.
- Pure 2h time stop.
- Standalone short.
- Pair RV short under strict cost.
- A7 cross-exchange live/shadow selector.
- Funding/OI standalone long or short.

## Promotion Thresholds

Do not promote based on the current sample.

Minimum forward-sample gates:

- Core P2 trades >= 100 for initial core basket evaluation.
- CP60 exits >= 50 for initial checkpoint evaluation.
- Protected exits >= 30 for initial Protect_A evaluation.
- Overflow trades >= 30 for initial O6 evaluation.
- Overflow trades >= 50 or protected exits >= 50 before any stronger upgrade discussion.

Current local paper-live is below all promotion thresholds.

## Current Bottlenecks

1. Forward sample insufficiency.
2. Live-loop continuity: local data was stale until manual refresh, so server paper-live/rsync/cron should be checked if continuous operation was expected.
3. Orderflow coverage: data cache refreshed, but reclaim-window coverage is still not usable for current CIC/P2 evaluation.
4. Token-level on-chain coverage: mapping coverage improved, but token-prior event overlap is still below attribution thresholds.
5. Real-live readiness: still blocked by sample size, execution realism, and risk envelope validation.

## Recommended Operating Mode

Freeze new broad alpha exploration for now.

Continue:

- P2 max8 core paper/shadow logging.
- O6 overflow shadow logging.
- CP60 and Protect_A cap2 counterfactual/shadow logging.
- Failure risk-off diagnostic logging.
- Low-coimpulse/router diagnostic logging.
- Orderflow/orderbook/on-chain/crowding fields in the forward ledger.

Run a weekly promotion audit only when sample thresholds are met. Until then, all short-window live results should be labeled:

- sample_status = insufficient
- evaluation_status = no_decision

Real-live remains disabled.
