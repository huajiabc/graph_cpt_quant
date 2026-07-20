# v14.9 FSS3 Forward-Shadow Deployment

Asset: `FSS3_CURRENT_SIGN_070_TURNOVER_CAP`

Transition: `frozen_forward_shadow_candidate -> LIVE_RECORD_ONLY`

Application:

- scope: `live_shadow`
- enabled: `true`
- push policy: `record_only`
- runner: `scripts/fss3_forward_shadow_once.py`
- real orders: disabled
- leverage: disabled

The frozen signal, beta hedge, gross normalization, 0.70 full-L1 transition
cap, and 20/40 bp cost shadows are unchanged. The application persists exact
decision inputs, hashes, executed weights, state lineage, virtual price/funding
PnL and explicit timeliness flags.

The latest saved v14.9 research weights are the initial state. Any decision
reconstructed after its natural Monday deadline is marked non-timely and cannot
count as forward evidence. PaperLive review remains blocked until new complete
natural weeks exist and the data/state/execution telemetry has been audited.

No order client, push route, leverage path, exchange credential or automatic
trade permission is present in this application.

## Remote verification

- host project: `E:\graph_quant`
- first isolated refresh: 2026-07-17 08:41 UTC, 72/72 symbols, zero fetch errors
- state catch-up decision: 2026-07-13 00:00 UTC, explicitly non-timely
- catch-up breadth: 10 negative-funding names and 61 positive-funding names
- executed weights: 72, gross notional 1.0, transition turnover 0.70
- cap breach: 0.0
- residual estimated BTC beta: approximately zero
- automatic loop verification: 2026-07-17 08:51 UTC, `fss3_forward_shadow rc=0`
- pre-deployment loop backup:
  `E:\graph_quant\logs\deploy_backups\fss3_20260717_164057`

The catch-up week is diagnostic state continuity only. Its return is excluded
from all natural-forward review counts. The first eligible decision can only be
created by the post-deployment Monday loop.
