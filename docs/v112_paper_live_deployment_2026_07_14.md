# v11.2 Topology PaperLive Deployment (2026-07-14)

## Deployment result

Remote host `DESKTOP-MDOT7C3` now runs the v11.2 topology observer inside the existing
`GraphQuant_PaperLive_Loop` scheduled task. The first automatic cycle completed:

- `v07d2 rc=0`
- `v10_short_s1_diagnostic rc=0`
- `v112_topology_shadow rc=0`
- `health_v07d2 rc=0`

The scheduled task remains running. v11.2 is `paper_live_shadow_only`; its config and every signal
row set `real_orders_allowed=false`. The module has market-data collection and virtual-return
accounting only, with no order route.

## Universe handling

The retrospective v11.2 universe contains BTC plus 72 community members. On the remote Bybit feed,
`TONUSDT` stopped producing bars at 2026-06-15 09:00Z and has only 346 admissible hourly history
rows for the July graph, below the frozen 500-row minimum.

Two states are preserved explicitly:

- Exact v11.2: `exact_universe_ready=false`; it is not generating forward evidence.
- Live adaptation: `active_universe_mode=adapted_fallback`, using the month-start trailing-30-day
  turnover rule. The July replacement is `TONUSDT -> JTOUSDT`.

The adaptation reconstructs 72 members as eight communities of nine. It is a separate shadow
observation and must not be merged into the exact v11.2 forward verdict.

## Frozen live parameters

- Graph history: 30 days; minimum 500 hourly observations.
- Communities: deterministic recursive spectral bisection, 8 x 9.
- Break state: 12-hour coherence crosses below the historical fifth percentile.
- Severity gate: expanding prior-event 80th percentile; current threshold 0.1037434363 from 955
  historical base events.
- Market state: BTC 24-hour volatility above the month-frozen 75th percentile; current threshold
  0.0059249825.
- Virtual horizon: 4 hours; round-trip cost: 20 bp.
- First-seen timeliness: 60 minutes. Historical backfills cannot become timely observations.

## Remote state after the automatic cycle

- Status: `READY_ADAPTED_SHADOW`.
- Latest source feature: 2026-07-14 12:00:00Z; data stale: false.
- Current-month base events: 62.
- Selected current signals: 0.
- Cumulative timely signals: 0.
- Virtual portfolios: 0.
- Blocking reasons: none.
- Warnings: exact universe unavailable; adapted universe is not exact v11.2 forward evidence.

## Files and rollback

- Config: `configs/v11_2_high_vol_topology_paper_live.yaml`
- Runner: `scripts/v112_topology_live_once.py`
- Observer: `src/pressure_graph/paper_live/v112.py`
- Remote status: `reports/v11_2_high_vol_topology_paper_live/live_status.json`
- Remote ledgers: `reports/v11_2_high_vol_topology_paper_live/forward/`
- Remote loop backup: `scripts/deploy_backups/20260714T115137Z/`
- Invalid 71-member bootstrap run was preserved at
  `reports/v11_2_invalid_exact_bootstrap_20260714T115731Z/` and is not part of canonical ledgers.

## Verification

- Local Ruff checks: passed.
- Local full suite: 451 passed, with four existing NumPy constant-series warnings.
- PowerShell loop parser: passed.
- Remote hashes/import/config seed checks: passed.
- Automatic v11.2 cycle: return code 0.
