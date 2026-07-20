# v10.1a Continuous BTC Attribution Correction - Pre-Registration

The v10.1 BTC source was event-windowed and covered only 49/81 OF1 candidates.
This correction changes only the BTC benchmark source:

- source: `data/raw/bybit/klines/BTCUSDT.parquet`;
- interval: continuous 15-minute Bybit BTCUSDT bars;
- benchmark entry: first 15m bar open at or after the token signal, with a
  maximum 15-minute alignment tolerance;
- benchmark exit: first 15m bar open at or after token signal +240 minutes;
- token signals, exact-flow states, token entries/exits, costs, splits, random
  seed, and all raw-long results remain unchanged.

The corrected attribution reports full and split token-minus-BTC gross return
and 40bp two-leg hedged net return. It cannot promote a strategy. A meaningful
idiosyncratic clue requires at least 90% BTC coverage, non-negative validation
and holdout hedged net40, and a matched-random percentile of at least 90%.
