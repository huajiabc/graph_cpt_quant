from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd

from pressure_graph.binance_metrics_history import (
    BinanceMetricsConfig,
    _merge_existing_metrics,
    inventory_binance_metrics,
    metrics_daily_url,
    parse_metrics_zip,
    write_binance_metrics_inventory,
)


def _archive(frame: pd.DataFrame) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSDT-metrics-2026-06-01.csv", frame.to_csv(index=False))
    return payload.getvalue()


def test_metrics_daily_url() -> None:
    assert metrics_daily_url("BTCUSDT", date(2026, 6, 1)).endswith(
        "/BTCUSDT/BTCUSDT-metrics-2026-06-01.zip"
    )


def test_parse_metrics_zip_adds_canonical_symbol() -> None:
    source = pd.DataFrame(
        {
            "create_time": ["2026-06-01 00:05:00"],
            "symbol": ["1000SHIBUSDT"],
            "sum_open_interest": [10.0],
            "sum_open_interest_value": [20.0],
            "count_toptrader_long_short_ratio": [1.1],
            "sum_toptrader_long_short_ratio": [1.2],
            "count_long_short_ratio": [1.3],
            "sum_taker_long_short_vol_ratio": [1.4],
        }
    )
    parsed = parse_metrics_zip(_archive(source), "SHIB1000USDT")
    assert parsed.loc[0, "bybit_symbol"] == "SHIB1000USDT"
    assert parsed.loc[0, "binance_symbol"] == "1000SHIBUSDT"
    assert str(parsed.loc[0, "create_time"].tz) == "UTC"


def test_incremental_metrics_merge_preserves_history(tmp_path) -> None:
    path = tmp_path / "BTCUSDT.parquet"
    old = pd.DataFrame(
        {
            "bybit_symbol": ["BTCUSDT"],
            "create_time": [pd.Timestamp("2026-06-01", tz="UTC")],
            "value": [1.0],
        }
    )
    old.to_parquet(path, index=False)
    new = pd.DataFrame(
        {
            "bybit_symbol": ["BTCUSDT", "BTCUSDT"],
            "create_time": [
                pd.Timestamp("2026-06-01", tz="UTC"),
                pd.Timestamp("2026-06-02", tz="UTC"),
            ],
            "value": [2.0, 3.0],
        }
    )
    merged = _merge_existing_metrics(new, path)
    assert merged["value"].tolist() == [2.0, 3.0]


def test_inventory_preserves_all_local_files_and_expected_missing_symbol(
    tmp_path,
) -> None:
    for symbol in ("BTCUSDT", "ETHUSDT"):
        frame = pd.DataFrame(
            {
                "create_time": pd.to_datetime(
                    ["2026-06-01 00:05:00", "2026-06-02 00:05:00"], utc=True
                ),
                "binance_symbol": [symbol, symbol],
                "bybit_symbol": [symbol, symbol],
                "source_day": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            }
        )
        frame.to_parquet(tmp_path / f"{symbol}.parquet", index=False)
    inventory = inventory_binance_metrics(
        tmp_path, expected_symbols=["BTCUSDT", "ETHUSDT", "MNTUSDT"]
    )
    assert set(inventory["bybit_symbol"]) == {"BTCUSDT", "ETHUSDT", "MNTUSDT"}
    assert inventory.loc[inventory["bybit_symbol"].eq("BTCUSDT"), "rows"].iloc[0] == 2
    assert inventory.loc[inventory["bybit_symbol"].eq("MNTUSDT"), "error"].iloc[0] == "missing_data_file"

    written = write_binance_metrics_inventory(
        BinanceMetricsConfig(output_root=tmp_path),
        expected_symbols=["BTCUSDT", "ETHUSDT", "MNTUSDT"],
    )
    assert len(written) == 3
    assert len(pd.read_csv(tmp_path / "manifest.csv")) == 3
