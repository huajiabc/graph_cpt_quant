# v10.1 Exact-Flow Persistence Findings

Status: `reject_exact_flow_240m_persistence`. Post-discovery offline audit;
no live or leverage permission changed.

The frozen OF1 state produced +0.4188% mean raw net20 at 240 minutes across 81
events. Development, validation, and holdout were +0.9096%, -0.0685%, and
+0.1243%. It failed the frozen validation, holdout-count, bootstrap, and
same-symbol/same-day random-time gates.

- day-block bootstrap 95% interval: -0.1717% to +1.0563%;
- random-time percentile: 84.8%, below the required 90%;
- full net30: +0.3188%;
- +60-minute shifted timing was weaker by 0.4885%;
- maximum positive-day contribution: 31.22%.

The combined result is not validated persistence alpha. Path diagnostics show
that the weakness is concentrated in `momentum_ignition`; the
`short_squeeze + OF1` subset is positive in all three chronological segments,
but this path was identified after reading v10.1 and needs its own
multiple-discovery-aware control.

The first BTC attribution used the event-window 1m execution-public file and
covered only 49/81 candidates, including only six validation events. Its
hedged result is therefore descriptive, not a reliable beta verdict. v10.1a
repeats only the attribution using the pre-existing continuous Bybit 15m BTC
series; it cannot alter raw-long or timing-control results.
