# v14.6 Exact v11.2 Temporal-Extension Preregistration

Date frozen: 2026-07-15, before rebuilding v11.2 or inspecting any post-canonical event/return.

## Purpose and immutable strategy

v11.2 is the strongest prior graph-specific price-behavior result, but it had only 43 four-hour
events. v14.6 does not alter it. It applies the exact frozen balanced-community topology-break,
sparse severity, BTC 24-hour high-volatility gate, community ranking, costs, horizons, cooldown,
random partitions, and delayed-signal control to the newly available Bybit price tail through
2026-07-14.

## Data splice and parity

- Keep every canonical v10.8/v11.2 panel row through its last timestamp unchanged.
- Construct only later rows from the combined historical/recent closed Bybit 15-minute prices,
  restricted to the canonical symbol universe. Compute trailing one-hour and future four-hour
  simple returns on the regular 15-minute matrix.
- Rebuild the complete strategy. Every canonical portfolio key
  `(candidate, feature_time, horizon_hours)` must be recovered, and gross/net return maximum
  absolute difference must be at most `1e-12`. Any parity failure invalidates the extension.

## Frozen primary confirmation

The primary horizon remains four hours. New evidence means entries strictly after the canonical
last entry (`2026-05-11 14:00 UTC`). Continue the existing shadow only if:

- at least 10 new four-hour observations span at least two calendar months;
- new mean net20 and net50 are positive;
- the new entry-day block-bootstrap 95% lower bound for net20 is positive;
- the full extended four-hour result remains at or above the 95th percentile of the newly rebuilt
  50-partition random family and beats the 24-hour delayed signal;
- canonical parity passes.

All other horizons are diagnostics. This test cannot grant PaperLive, leverage, or live-order
permission; it can only continue or weaken the existing forward-shadow rationale.
