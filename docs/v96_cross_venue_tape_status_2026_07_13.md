# v9.6 Cross-Venue Tape Deployment Status (2026-07-13)

## Verdict

- Alpha status: `DATA_ACCUMULATING`.
- Trading permission: none. The collector is data-only and is not imported by the paper-live or order-routing path.
- Earliest admissible research minute: `2026-07-13T11:01:00Z`.
- The earlier `2026-07-13T10:58:00Z` startup minute is excluded from research and latency QA because the first implementation measured flush delay instead of event receipt delay.

## Frozen initial universe

`BTCUSDT, ETHUSDT, SOLUSDT, HYPEUSDT, XRPUSDT, ZECUSDT, DOGEUSDT, NEARUSDT, SUIUSDT, ONDOUSDT, 1000PEPEUSDT, XLMUSDT, XAUTUSDT, ADAUSDT, WLDUSDT, TAOUSDT, FARTCOINUSDT, LINKUSDT, BNBUSDT, ENAUSDT`

The universe was selected once from the remote live feature ranking, intersected with Binance USDT instruments, with BTC/ETH/SOL forced into the core set. It must not be retrospectively changed for the first evaluation window.

## Remote deployment

- Scheduled task: `GraphQuant_CrossVenue_Tape`.
- Principal: user `o`, interactive logon, limited run level.
- Trigger: at startup; also started manually after registration.
- Multiple instances: `IgnoreNew`.
- Execution time limit: none (`PT0S`).
- Task-level restart count: 999 with one-minute interval; the wrapper also restarts a failed recorder after ten seconds.
- Recorder process: one logical recorder. Windows exposes the venv launcher and its base-Python child as two Python process records.
- Existing task `GraphQuant_PaperLive_Loop` was not stopped, reconfigured, or reloaded.

## First admissible minute QA

- Minute: `2026-07-13T11:01:00Z`.
- Combined rows: 40.
- Binance rows: 20.
- Bybit rows: 20.
- Synchronized symbols: 20 / 20.
- Maximum receipt-to-event lag: 4.586992 seconds.
- P95 receipt-to-event lag: 4.414188 seconds.
- Rows above the 10-second stale threshold: 0.
- The Binance socket's first opening handshake timed out once; the one-second reconnect succeeded and emitted a first-trade audit event.

## Paper-live isolation check

At `2026-07-13T10:51:36Z`, the existing paper-live health report showed `api_probe_ok: True`, `data_stale: False`, and feature time `2026-07-13T10:45:00Z`. The subsequent v07d2, short diagnostic, health, v08 orderflow, and v085 orderbook jobs all completed with `rc=0`.

## Evaluation gate

No alpha verdict is allowed before all preregistered gates in `v96_cross_venue_buy_pressure_prereg_2026_07_13.md` are satisfied: at least 90 calendar days, three distinct months, 200 entries, adequate monthly breadth, synchronized coverage of at least 95%, cost sensitivity, chronological holdout, bootstrap stability, concentration limits, and placebo/control separation.

## 2026-07-15 stall incident and recovery

The scheduled task still reported `Running`, but the last fragment had stopped at
`2026-07-13T14:47:00Z`. Only about three hours of tape existed and the following
gap is unavailable for research:

`2026-07-13T14:48:00Z` through `2026-07-15T07:53:59Z`.

The recorder had two coupled failure modes. Its one-minute flush coroutine
synchronously rebuilt the full coverage report on the asyncio event loop, which
blocked WebSocket heartbeats as the fragment inventory grew. A Bybit ping-task
exception could then escape cleanup and terminate the venue loop. Fragment reads
also scanned every day before applying the requested time range.

The deployed repair moved coverage computation to a worker thread, reduced its
default refresh frequency to five minutes, suppressed ping cleanup exceptions,
added explicit `bar_flush` heartbeats, and pruned fragment day directories before
reading. The original remote files were backed up under
`scripts/deploy_backups/20260715T155238_cross_venue_recovery` and replaced only
after hash checks and `py_compile` passed.

Both venue sockets reconnected at approximately `2026-07-15T07:53:09Z`. The first
new complete minute was `07:54Z` with 40 rows, followed by another 40-row flush at
`07:55Z`. `GraphQuant_CrossVenue_Tape` and `GraphQuant_PaperLive_Loop` both
remained `Running`; PaperLive was not restarted or reconfigured. Accumulation
continues from `07:54Z`, but the missing interval cannot be backfilled or counted
toward the 90-day gate.
