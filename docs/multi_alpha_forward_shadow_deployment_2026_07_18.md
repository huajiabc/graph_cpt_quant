# Multi-Alpha Forward-Shadow Deployment

Date: 2026-07-18.

This deployment adds the remaining defensible candidates to the remote
GraphQuant observation loop without creating any real-order, push, or leverage
path. The three additions deliberately have different lifecycle semantics.

## CM2 fixed core-satellite

- Asset: strategy `CM2_FIXED_80_FSS3_20_TG1`.
- Strategy transition:
  `CANDIDATE_WATCH -> FROZEN_CORE -> LIVE_RECORD_ONLY`.
- Application transition:
  `candidate -> live_shadow`, `enabled=true`,
  `push_policy=record_only`.
- Runner: `scripts/cm2_forward_shadow_once.py`.
- Config: `configs/v16_5_cm2_forward_shadow.yaml`.
- Exact construction: 80% independently running FSS3 and 20% independently
  running TG1, with no dynamic allocation or cross-sleeve netting.
- TG1 remains `REFERENCE_ONLY` as a standalone strategy. It is executed only
  as an isolated internal CM2 sleeve.
- A missing or ambiguous FSS3/TG1 state fails the complete CM2 week closed.
- The historical v16.5 promotion gates and v16.6 independent audit passed, but
  at least 12 new natural forward weeks are still required before review.

## q90 book-vacuum OCO

- Asset: strategy
  `DVB5_POSITIVE_PRESSURE_0625SIGMA_BTC_BREAKOUT`.
- Strategy status remains `CANDIDATE_WATCH`; it is not statistically
  confirmed.
- Application transition:
  `candidate -> live_shadow`, `enabled=true`,
  `push_policy=record_only`.
- Runner: `scripts/q90_forward_shadow_once.py`.
- Config: `configs/v23_8_q90_forward_shadow.yaml`.
- Frozen rule: positive q90 cross-coin book pressure, 0.625-sigma BTC OCO,
  four-hour horizon, 10/20 bp primary/stress cost.
- Binance book-depth archives are published after the represented market
  hours. The ledger is valid for untouched forward-research evidence but is
  permanently marked `timely_execution_eligible=false`.
- Virtual outcomes are loaded only after a frozen positive event exists and
  its four-hour horizon has elapsed.

## Liquidation graph bucket

- Asset: factor `LIQUIDATION_GRAPH_BUCKET_VOLATILITY_STATE`.
- Factor transition: `FROZEN_FACTOR -> LIVE_DIAGNOSTIC`.
- This is not registered as a strategy.
- Runner: `scripts/liquidation_graph_forward_once.py`.
- Config:
  `configs/v23_42_liquidation_graph_live_diagnostic.yaml`.
- The remote causal start is the first successful remote batch completion.
  The retrospective endpoint snapshot is never backdated.
- Only 5/15/60-minute graph features are appended. Outcomes stay unloaded
  until the frozen gate of 336 hourly decisions across at least 14 UTC days.

## Permission boundary

All three applications use isolated data/report roots. Every config explicitly
sets `real_orders_allowed=false` and `leverage_allowed=false`. CM2 and q90 use
`scope=live_shadow` with `push_policy=record_only`; the liquidation graph is a
factor diagnostic only. No order file, exchange-authenticated endpoint, order
router, or automatic alert/push path is introduced.

## Verification record

- Local regression: `754 passed`, with four pre-existing NumPy constant-series
  warnings.
- The standalone q90 implementation reproduced all 159 frozen historical
  events: entry times and every numeric feature were identical to the frozen
  research output.
- The standalone liquidation feature builder matched the frozen v23.35
  contract exactly on real collected events; maximum absolute feature
  difference was zero.
- Deployment archive:
  SHA-256
  `B67B45DD174D9FB1C5E2D96D82928096AE3D1EDE7AB3F415CE044114C3719946`.
- Remote pre-deployment backup:
  `E:\graph_quant\logs\deploy_backups\multi_alpha_20260718_144209`.
- CM2 first remote run completed with exit code zero and status
  `READY_RECORD_ONLY`. It recorded one 2026-07-13 catch-up decision with nine
  TG1 names and one aligned CM2 week. The decision is explicitly non-timely
  and not evidence-eligible, so completed natural forward evidence remains
  zero weeks.
- q90 first remote run completed with 48 contiguous delayed-forward hours and
  zero positive q90 events. The next scheduled cycle successfully collected
  the 2026-07-17 official archive, extending the ledger to 72 contiguous
  delayed-forward hours; the frozen rule still produced zero positive events.
- The liquidation graph first remote batch completed for 17 symbols, passed
  all eight source-audit checks, and froze the remote causal start at
  `2026-07-18T06:49:46.489508Z`.
- The main `GraphQuant_PaperLive_Loop` scheduled task was restored after the
  first-run checks. Its first restored cycle completed CM2, q90, and the
  liquidation diagnostic with exit code zero at 07:30:36Z, 07:31:09Z, and
  07:31:24Z, respectively. The liquidation run appended the first causal
  hourly ledger row and brought the source total to 3,602 events. The
  temporary CM2 first-run task was removed.
