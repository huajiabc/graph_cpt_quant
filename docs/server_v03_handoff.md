# v0.3 Server Handoff

## Goal

Run `v0.3 All-Eligible Dynamic Universe Revalidation` on a server without changing C2 rules.

C2 remains frozen:

- path: Short Squeeze
- entry: `E4_pullback_0.5pct_valid_4_bars`
- execution: `swing`
- costs: `5/10/20/30/50` bps single-side

## Recommended Server

- Python 3.10
- 8+ vCPU
- 64GB RAM minimum
- 128GB RAM recommended
- 100GB+ free NVMe disk

The local crash happened during pandas full-table report preparation on a 12,895,088 row feature table. The current code path has been lightened for C2-only v0.3, but 128GB is still the calmer choice.

## Minimal Files To Transfer

For report-only continuation:

- project source: `configs/`, `src/`, `tests/`, `pyproject.toml`, `README.md`
- `data/processed/v0_3/perp_pressure_features_all_eligible.parquet`
- `data/raw/bybit/instruments.parquet`

Optional, only if rebuilding features on the server:

- `data/raw/bybit/klines/*.parquet`
- `data/raw/bybit/funding/*.parquet`
- `data/raw/bybit/open_interest/*.parquet`
- `data/raw/bybit/tickers.parquet`

## Commands

```bash
cd /path/to/graph_quant
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest -q
python -m ruff check .

pressure-graph run-v03 --config configs/v0_3.yaml
```

Expected outputs:

- `reports/v0_3/c2_all_eligible_summary.csv`
- `reports/v0_3/c2_ex_may_summary.csv`
- `reports/v0_3/c2_monthly_contribution.csv`
- `reports/v0_3/c2_symbol_contribution.csv`
- `reports/v0_3/c2_liquidity_bucket.csv`
- `reports/v0_3/c2_universe_topn_compare.csv`
- `reports/v0_3/c2_matched_baseline.csv`
- `reports/v0_3/c2_entry_only_baseline.csv`
- `reports/v0_3/c2_leave_one_month_out.csv`
- `reports/v0_3/c2_portfolio_concurrency.csv`
- `reports/v0_3/candidate_list.md`

## If Rebuilding Features

```bash
pressure-graph build-v03-features --config configs/v0_3.yaml
pressure-graph run-v03 --config configs/v0_3.yaml
```

The raw all-eligible Bybit snapshot currently has 368 symbols with kline/funding/OI coverage.

