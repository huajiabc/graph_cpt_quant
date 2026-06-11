# Crypto Pressure Graph v0

Contract pressure path experiment for USDT perpetuals.

The v0 question is narrow: can price, volume, settled funding, and open interest identify stable long-side pressure paths that improve future 4h/12h upside hit rates without worsening pre-hit drawdown?

## Setup

```powershell
python -m pip install -e ".[dev]"
```

## Main Commands

```powershell
pressure-graph collect --exchange bybit --days 7 --symbols BTCUSDT,ETHUSDT
pressure-graph build-features --exchanges bybit
pressure-graph run-paths --universe-col universe_static_current_top30
```

Binance recent OI smoke:

```powershell
pressure-graph run-all --exchange binance --days 30 --symbols BTCUSDT,ETHUSDT --universe-col universe_static_current_top30
```

Bybit 12-month main run bootstrap:

```powershell
pressure-graph run-all --exchange bybit --days 365 --all-eligible --universe-col universe_dynamic_monthly_top30
```

The full Bybit run can issue many thousands of requests because historical OI is paginated at a small limit.

## Data Contract

Primary key:

```text
exchange, symbol, bar_open_time
```

Time fields:

```text
bar_open_time   15m candle start time
bar_close_time  bar_open_time + 15m
feature_time    bar_close_time
entry_time      next bar_open_time
```

Funding is backward as-of joined using only settled records:

```text
funding_rate_settled, funding_time, funding_age_minutes, funding_interval_minutes
```

Open interest is stored in both base and USDT value terms:

```text
oi_base, oi_value_usdt, oi_value_delta_1h/4h, oi_value_delta_1h/4h_percentile
```

Rolling percentiles are current value versus the prior rolling window. Future labels start from the next bar, never the current bar.

## Outputs

```text
data/processed/v0/perp_pressure_features.parquet
reports/v0/path_stats.csv
reports/v0/development_walk_forward.csv
reports/v0/final_holdout.csv
reports/v0/parameter_grid_stats.csv
reports/v0/param_heatmaps/*.png
reports/v0/candidate_list.md
```

`data/` and `reports/` are ignored by git because they are generated artifacts.
