# v18.4 BTC-Inclusive Binance Metrics Audit

Verdict: `audit_pass_btc_inclusive_metrics_ready`.

Checks: 14; passed: 14.

BTC rows: 109,146; source days: 379; five-minute grid coverage: 99.995419%.

Exact 15-minute panel symbols: 46; bars: 32,460; bars with >=40 taker symbols: 32,417.

No forward fill is used. A metric is available only when its archived
timestamp exactly equals the completed 15-minute price-bar close.
No live, PaperLive, application, leverage, remote, or order scope changed.
