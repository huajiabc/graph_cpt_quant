from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v41_cross_exchange_lead_lag_diagnostics import (
    V41Config,
    write_v41_cross_exchange_lead_lag_diagnostics,
)


def _write_klines(root: Path, exchange: str, symbol: str, closes: list[float], volumes: list[float]) -> None:
    times = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    prev = [closes[0], *closes[:-1]]
    rows = []
    for idx, close in enumerate(closes):
        open_price = prev[idx]
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "bar_open_time": times[idx],
                "bar_close_time": times[idx] + pd.Timedelta(minutes=15),
                "open": open_price,
                "high": max(open_price, close) * 1.002,
                "low": min(open_price, close) * 0.998,
                "close": close,
                "volume": volumes[idx],
                "turnover": volumes[idx] * close,
                "trades": 100 + idx,
                "taker_buy_base": volumes[idx] * (0.7 if exchange == "binance" and idx in {5, 6} else 0.45),
                "taker_buy_quote": volumes[idx] * close * (0.7 if exchange == "binance" and idx in {5, 6} else 0.45),
            }
        )
    out = root / "raw" / exchange / "klines"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / f"{symbol}.parquet", index=False)


def test_v41_writes_response_and_incremental_tables(tmp_path: Path) -> None:
    vols = [900.0, 1000.0, 1100.0, 950.0, 1050.0, 10000.0, 8000.0, 1020.0, 990.0, 1010.0, 970.0, 1030.0]
    flat = [900.0, 1000.0, 1100.0, 950.0, 1050.0, 1000.0, 980.0, 1020.0, 990.0, 1010.0, 970.0, 1030.0]
    _write_klines(tmp_path, "binance", "AAAUSDT", [100, 100, 100, 100, 100, 103, 103.5, 103.4, 103.8, 104.0, 104.2, 104.4], vols)
    _write_klines(tmp_path, "bybit", "AAAUSDT", [100, 100, 100, 100, 100, 99.5, 101, 102, 103, 104, 105, 105.5], flat)
    _write_klines(tmp_path, "binance", "BBBUSDT", [50, 50, 50, 50, 50, 50.1, 50, 50.1, 50, 50.1, 50, 50.1], flat)
    _write_klines(tmp_path, "bybit", "BBBUSDT", [50, 50, 50, 50, 50, 50, 50.1, 50.0, 50.1, 50, 50.1, 50], flat)

    outputs = write_v41_cross_exchange_lead_lag_diagnostics(
        V41Config(report_root=tmp_path / "reports", data_root=tmp_path, top_n=2)
    )

    response = pd.read_csv(outputs["response_curve"])
    evaluated = response[response["status"].eq("evaluated_15m_proxy")]
    assert not evaluated.empty
    assert evaluated["events"].max() > 0
    incremental = pd.read_csv(outputs["incremental_edge_vs_target_baseline"])
    assert "LX1_source_prior_target_lag_reclaim" in set(incremental["candidate"])
    coverage = pd.read_csv(outputs["timeframe_coverage_audit"])
    assert set(coverage["dataset"]) == {"source_15m", "target_15m", "source_1m", "target_1m"}
