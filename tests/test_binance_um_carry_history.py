from __future__ import annotations

import io
import zipfile

import pandas as pd

from pressure_graph.binance_um_carry_history import (
    parse_funding_payload,
    parse_um_kline_zip,
    um_symbol,
)


def _archive() -> bytes:
    row = [
        1_754_006_400_000,
        100.0,
        102.0,
        99.0,
        101.0,
        10.0,
        1_754_009_999_999,
        1000.0,
        42,
        4.0,
        400.0,
        0,
    ]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("1000SHIBUSDT-1h.csv", ",".join(map(str, row)) + "\n")
    return payload.getvalue()


def test_um_alias_and_kline_closed_time() -> None:
    assert um_symbol("SHIB1000USDT") == "1000SHIBUSDT"
    frame = parse_um_kline_zip(_archive(), "SHIB1000USDT")
    assert frame.loc[0, "binance_symbol"] == "1000SHIBUSDT"
    assert frame.loc[0, "feature_time"] == pd.Timestamp(
        "2025-08-01 01:00:00", tz="UTC"
    )


def test_funding_millisecond_offset_is_normalized_to_scheduled_second() -> None:
    payload = [
        {
            "symbol": "BTCUSDT",
            "fundingTime": 1_754_006_400_001,
            "fundingRate": "-0.00001408",
        }
    ]
    frame = parse_funding_payload(payload, "BTCUSDT", "BTCUSDT")
    assert frame.loc[0, "funding_time"] == pd.Timestamp("2025-08-01", tz="UTC")
    assert frame.loc[0, "funding_rate_settled"] == -0.00001408
