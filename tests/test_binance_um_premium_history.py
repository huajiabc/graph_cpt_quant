from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date

import pandas as pd

from pressure_graph.binance_um_premium_history import (
    BinancePremiumConfig,
    _merge_existing,
    daily_premium_url,
    inventory_binance_premium,
    monthly_premium_url,
    parse_premium_zip,
    verify_archive_checksum,
    write_premium_inventory,
)


def _archive(rows: list[list[object]]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        frame = pd.DataFrame(rows)
        archive.writestr(
            "BTCUSDT-15m-2026-06.csv",
            frame.to_csv(index=False, header=False),
        )
    return payload.getvalue()


def test_premium_urls() -> None:
    assert monthly_premium_url("BTCUSDT", date(2026, 6, 1)).endswith(
        "/BTCUSDT/15m/BTCUSDT-15m-2026-06.zip"
    )
    assert daily_premium_url("BTCUSDT", date(2026, 6, 1)).endswith(
        "/BTCUSDT/15m/BTCUSDT-15m-2026-06-01.zip"
    )


def test_checksum_verification() -> None:
    content = b"premium archive"
    digest = hashlib.sha256(content).hexdigest()
    assert verify_archive_checksum(content, f"{digest}  file.zip".encode())
    assert not verify_archive_checksum(content, f"{'0' * 64}  file.zip".encode())


def test_parse_premium_zip_uses_completed_time() -> None:
    content = _archive(
        [
            [
                1_780_272_000_000,
                -0.0001,
                0.0002,
                -0.0003,
                0.0001,
                0,
                1_780_272_899_999,
                0,
                0,
                0,
                0,
                0,
            ]
        ]
    )
    parsed = parse_premium_zip(
        content, "BTCUSDT", "BTCUSDT", "BTCUSDT-15m-2026-06.zip"
    )
    assert parsed.loc[0, "feature_time"] == parsed.loc[0, "bar_open_time"] + pd.Timedelta(
        minutes=15
    )
    assert parsed.loc[0, "close"] == 0.0001
    assert str(parsed.loc[0, "feature_time"].tz) == "UTC"


def test_incremental_merge_and_inventory_preserve_expected_symbols(tmp_path) -> None:
    path = tmp_path / "BTCUSDT.parquet"
    old = pd.DataFrame(
        {
            "bybit_symbol": ["BTCUSDT"],
            "binance_symbol": ["BTCUSDT"],
            "source_archive": ["old.zip"],
            "feature_time": [pd.Timestamp("2026-06-01", tz="UTC")],
            "close": [0.1],
        }
    )
    old.to_parquet(path, index=False)
    new = old.copy()
    new["close"] = 0.2
    merged = _merge_existing(new, path)
    assert merged["close"].tolist() == [0.2]
    inventory = inventory_binance_premium(
        tmp_path, expected_symbols=["BTCUSDT", "ETHUSDT"]
    )
    assert set(inventory["bybit_symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert inventory.loc[inventory["bybit_symbol"].eq("ETHUSDT"), "error"].iloc[0] == "missing_data_file"
    written = write_premium_inventory(
        BinancePremiumConfig(output_root=tmp_path),
        expected_symbols=["BTCUSDT", "ETHUSDT"],
    )
    assert len(written) == 2
