# v17.2 Deribit Historical Option-Surface Feasibility

Status: `DATA_SOURCE_ACCEPTED_FOR_SIGNAL_RESEARCH_ONLY`

## Question

Can archived Deribit BTC options provide a longer and cleaner volatility-surface
signal than the short Binance EOHSummary sample?

## Official endpoint audit

- `public/get_instruments?expired=true` exposes only recently expired contracts,
  so it cannot enumerate a multi-year archive.
- Exact archived names remain queryable through `public/get_instrument`.
- `public/get_tradingview_chart_data` returns historical option OHLCV and cost
  bars for exact archived names.
- Zero-volume hours can contain exchange-filled prices. They are never treated as
  observations here. A usable bar must have both `volume > 0` and `cost > 0`.
- `public/get_mark_price_history` did not accept the archived contracts tested and,
  per Deribit documentation, covers only a subset of volatility-index options.

Official references:

- https://docs.deribit.com/api-reference/market-data/public-get-instruments
- https://docs.deribit.com/api-reference/market-data/public-get-instrument
- https://docs.deribit.com/api-reference/market-data/public-get-tradingview-chart-data
- https://docs.deribit.com/api-reference/market-data/public-get-mark-price-history

## Frozen collection rule

- BTC quarterly expiries: last Friday of March, June, September, and December at
  08:00 UTC.
- Window: 45 to 0 days before expiry; surface use is restricted to 7--45 DTE.
- Strike lattice: USD 5,000 spacing when reference BTC is below USD 40,000 and
  USD 10,000 otherwise, spanning approximately 0.6--1.4 times reference spot.
- Price: hourly option trade VWAP, exactly `cost / volume`.
- Underlying: same-hour midpoint of Deribit BTC perpetual open and close.
- Volatility: zero-rate inverse BTC Black-Scholes implied volatility.
- Signal timestamp: the next UTC midnight after the completed trade day.

## Observed coverage

- Queried contracts: 272.
- Contracts with actual positive-volume bars: 264.
- Active hourly trade bars: 74,962.
- Daily surface rows: 569; quality-passing rows: 567.
- Quarterly expiries: 22.
- Feature range: 2021-03-02 through 2026-06-15 UTC.
- Median daily cross section rises from six contracts in 2021 to thirteen in
  2025.

The reconstructed ATM IV has 558 overlapping days with official BTC DVOL. Its
correlation with daily DVOL is 0.9632 and its median absolute difference is
0.0284 volatility points in decimal units. This is a strong unit and timing
sanity check, not evidence of alpha.

## Boundary

This archive supports causal option-surface signals applied to futures or spot
returns. It does **not** reconstruct historical bids, asks, spreads, queue
position, or option execution. No historical option-PnL claim may use these
trade bars as if they were executable quotes.
