# v10.7 Cross-Venue Flow Graph Status

Date: 2026-07-14

Status: `DATA_ACCUMULATING` remotely and `DATA_UNAVAILABLE_LOCAL` in the current workspace.

## What was completed

- Frozen the cross-symbol Binance/Bybit flow-graph hypothesis and controls before outcomes exist.
- Added a fail-closed readiness pipeline that builds only synchronized, complete, non-stale
  symbol-minutes and audits every symbol-day.
- Added explicit gates for 90 calendar days, three months, 15 usable symbols, 95% synchronized
  coverage, and 80 passing full days per symbol.
- The pipeline cannot emit an alpha verdict unless all gates pass.

The current local run correctly produced zero synchronized rows and
`alpha_verdict_allowed=False`, because the remote tape is not mirrored locally. The previously
verified remote recorder began its first admissible minute at 2026-07-13 11:01:00Z. At the
current run time only about 0.89 calendar days had elapsed.

The earliest time-only gate is 2026-10-11 11:01:00Z. Missing coverage or sample breadth can move
the actual evaluation date later.

## Historical-data decision

Existing Bybit public-trade archives cover event-selected dates, not an unconditional continuous
panel. v10.0 caches are single-venue and inherit that event-day selection. v0.8 orderflow files
contain only minutes. None may be spliced into v10.7, because doing so would create selection
bias and violate the synchronized two-venue contract.

The UU remote window was discoverable, but its capture interface returned Windows error
`0x80004002`; UI input was stopped rather than operating without visual verification. No remote
task or configuration was changed.

No PaperLive, leverage, or live permission changed.
