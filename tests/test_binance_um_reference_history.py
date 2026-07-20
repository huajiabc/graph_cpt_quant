from datetime import date
from pathlib import Path

import pandas as pd

from pressure_graph.binance_um_reference_history import (
    BinanceReferenceConfig,
    daily_reference_url,
    inventory_binance_reference,
    monthly_reference_url,
)


def test_reference_urls() -> None:
    assert monthly_reference_url(
        "markPriceKlines", "BTCUSDT", date(2026, 5, 1)
    ).endswith(
        "/markPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2026-05.zip"
    )
    assert daily_reference_url(
        "indexPriceKlines", "BTCUSDT", date(2026, 6, 1)
    ).endswith(
        "/indexPriceKlines/BTCUSDT/15m/BTCUSDT-15m-2026-06-01.zip"
    )


def test_reference_inventory_preserves_missing_symbols(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "bybit_symbol": ["BTCUSDT"],
            "binance_symbol": ["BTCUSDT"],
            "source_archive": ["archive.zip"],
            "feature_time": [pd.Timestamp("2026-06-01", tz="UTC")],
        }
    )
    frame.to_parquet(tmp_path / "BTCUSDT.parquet", index=False)
    cfg = BinanceReferenceConfig(
        dataset="markPriceKlines", output_root=tmp_path
    )
    inventory = inventory_binance_reference(
        cfg, expected_symbols=["BTCUSDT", "ETHUSDT"]
    )
    assert set(inventory["bybit_symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert inventory.loc[inventory["bybit_symbol"].eq("ETHUSDT"), "error"].iloc[0] == "missing_data_file"
