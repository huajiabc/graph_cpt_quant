# New Alpha Exploration Findings — 2026-07-13

Status: first orthogonal-alpha round complete. No paper/live strategy, gate,
ranking, sizing, or permission was changed.

## 1. Token DEX attention -> CEX underreaction

Frozen candidate: `TAD1_UNDERREACTION_RECLAIM`.

- 87 trades across 16 symbols and 9 active months.
- net20 12h: -0.6014%.
- net30 12h: -0.8014%.
- entry-day block-bootstrap 95% CI: [-1.8141%, +0.3735%].
- validation: 4 trades, net20 -2.6801%.
- holdout: 11 trades, net20 +0.3776%.
- same-token/same-month random-time median net20: -0.4410%; real is
  below the random 90th percentile.
- P2 overlap: 0/87; the failure is independent of P2 overlap.

Verdict: `FAIL_HISTORICAL_SCREEN`. Reject the independent underreaction entry.
Do not relax the frozen thresholds.

## 2. Token DEX attention + CEX impulse confirmation

Frozen comparator/follow-up: `TAD2_CEX_CONFIRMATION`.

- 105 trades across 15 symbols and 10 active months.
- net20 12h: +0.3343%; net30 12h: +0.1343%.
- search / validation / holdout net20 are all positive, but validation and
  holdout contain only 10 and 5 trades.
- entry-day block-bootstrap 95% CI: [-0.8410%, +1.4164%].
- maximum month contribution: 105.50%.
- real beats same-token random-time p90 and same-chain random-token median.
- the event shifted forward seven days is much stronger: 41 trades, net20
  +3.3088%.
- removing the single P2-overlap trade leaves 104 trades, net20 +0.2399%.

Verdict: `FAIL_HISTORICAL_SCREEN`. The CEX momentum state has a weak positive
shape, but the DEX event-time attribution fails and concentration/uncertainty
remain unacceptable. Do not create a paper shadow from TAD2.

## 3. Continuous cross-exchange aggressor-flow propagation

The hypothesis remains structurally attractive, but it is not currently
backtestable from available data.

- Local: no `data/orderflow_history` continuous CVD shards.
- Remote: 49-symbol Bybit orderflow has accumulated from 2026-07-07 through
  2026-07-13 (about six days), but no matching Binance continuous aggTrades/CVD
  history exists.
- Existing Binance/Bybit 1-minute kline taker-buy proxies were already tested
  in v4 and did not isolate venue-specific alpha.

Verdict: `DATA_BLOCKED`. Do not substitute kline proxies for continuous
aggressor flow. A future collector must store matching Binance and Bybit 1m CVD
with immutable timestamps before this hypothesis is evaluated.

## 4. Listing washout / digestion

The v8 listing atlas is not usable for alpha inference with current raw data.

- 368 Bybit symbol kline files exist.
- The earliest file begins at least 157 hours after the recorded listing; the
  median delay is about 13,424 hours.
- Zero listing events have launch-adjacent raw coverage and a complete seven-day
  response window.
- The old v8 implementation fell back to the first local bar when the listing
  event predated raw coverage. Its reported 24h/7d listing returns therefore do
  not measure post-listing performance.

The v8 replay now fails closed with `event_time_not_covered`; the earlier
"7d digestion" clue is withdrawn.

## Conclusion and next priority

No new alpha passed this round. The useful output is a clean rejection of two
DEX-entry variants, removal of a misleading listing clue, and a precise data
contract for the only promising next orthogonal lane: synchronized continuous
Binance -> Bybit aggressor-flow propagation.

P2 remains the unchanged primary forward paper ledger. Real-live and canary
remain disabled.
