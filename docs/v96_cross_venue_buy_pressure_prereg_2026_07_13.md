# v9.6 Binance -> Bybit Aggressor-Flow Propagation Preregistration

Status: forward data collection and future research specification only. No
signal, order, paper portfolio, sizing rule, or live permission is changed.

## Data contract

- Source venue: Binance USDT perpetual public `aggTrade` WebSocket.
- Target venue: Bybit linear USDT perpetual public-trade WebSocket.
- Universe: recorder-start frozen Top20 common symbols by the existing Bybit
  live `dynamic_all_rank`; BTC, ETH, and SOL are retained when common.
- Storage: immutable one-minute aggregate fragments. Raw individual trades are
  not retained.
- Required fields: OHLC, trade count, buy/sell volume and turnover, CVD,
  imbalance, first/last event time and id, ingest time, and stream session id.
- Reconnect fragments for the same venue/symbol/minute are summed and their
  first/last prices are ordered by event time.
- A minute is research-eligible only when both venues have an observed complete
  bar for the same symbol and minute.
- Clock audit fails closed if either venue's last-event lag exceeds 10 seconds
  at ingestion or if a cross-venue minute is absent.

## Frozen hypothesis

`CVP1_BINANCE_BUY_PRESSURE_BYBIT_LAG`:

1. On complete synchronized bars, form a trailing five-minute Binance source
   window.
2. Source impulse requires:
   - five-minute turnover-weighted buy/sell imbalance >= +0.15;
   - five-minute Binance price return > 0;
   - five-minute turnover >= the strictly trailing seven-day, same-symbol 95th
     percentile. The current window is excluded from the percentile.
3. Bybit is lagging when its same five-minute return is <= +0.50% and the
   Binance-minus-Bybit five-minute return gap is >= +0.30%.
4. Confirmation is the first of the next three complete synchronized minutes
   where Bybit trailing-three-minute imbalance is >= +0.05 and Bybit close is
   above its close at the end of the source window.
5. Entry is the confirmation-minute close. Same-symbol signals use a frozen
   four-hour cooldown.
6. Outcomes: 1h, 4h, and 12h Bybit returns. Focal horizon is 4h; focal cost is
   20 bps per side, with 10/30/50 bps stress.

No threshold, horizon, universe, or cooldown may change after outcome data is
read. Any change is a new candidate/version.

## Frozen controls

- `TARGET_ONLY`: the exact Bybit confirmation state without Binance source
  conditions.
- `SOURCE_SHIFT_5M`: Binance source window shifted forward five minutes.
- `SAME_MINUTE_RANDOM_SYMBOL`: source event assigned to another common symbol.
- `REVERSE_VENUE`: Bybit source pressure -> Binance lag, evaluated as an
  attribution control rather than a tradable Binance strategy.
- Bearish propagation is logged symmetrically as a diagnostic; it cannot approve
  CVP1 or become an independent short strategy in this version.

## Sample and decision gate

No alpha verdict is allowed before all of the following are true:

- at least 90 calendar days spanning at least three calendar months;
- at least 200 eligible CVP1 entries, with at least 20 in each active month;
- synchronized-minute coverage >= 95% for every evaluated symbol-day;
- no single month or symbol contributes more than 35% of net20 alpha;
- search, validation, and holdout are chronological and disjoint;
- validation and holdout net20 at 4h are positive;
- entry-day block-bootstrap 95% lower bound is positive;
- real beats every frozen control and remains positive at 30 bps per side.

Until the gate is met, outputs are `DATA_ACCUMULATING` or `DATA_QUALITY_FAIL`,
never `alpha_pass`.
