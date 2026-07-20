from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd

from pressure_graph.binance_spot_history import (
    _month_starts,
    _partial_days,
    monthly_kline_url,
    parse_spot_kline_zip,
    spot_symbol,
)


def _archive(open_time: int, close_time: int) -> bytes:
    row = [
        open_time,
        100.0,
        102.0,
        99.0,
        101.0,
        10.0,
        close_time,
        1000.0,
        42,
        4.0,
        400.0,
        0,
    ]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSDT-1h.csv", ",".join(map(str, row)) + "\n")
    return payload.getvalue()


def test_spot_alias_and_url() -> None:
    assert spot_symbol("1000BONKUSDT") == "BONKUSDT"
    assert monthly_kline_url("BTCUSDT", date(2025, 8, 1)).endswith(
        "/BTCUSDT/1h/BTCUSDT-1h-2025-08.zip"
    )


def test_archive_schedule_uses_months_then_partial_days() -> None:
    assert _month_starts(date(2025, 8, 1), date(2025, 10, 4)) == [
        date(2025, 8, 1),
        date(2025, 9, 1),
    ]
    assert _partial_days(date(2025, 8, 1), date(2025, 10, 4)) == [
        date(2025, 10, 1),
        date(2025, 10, 2),
        date(2025, 10, 3),
        date(2025, 10, 4),
    ]


def test_parse_spot_archive_handles_microsecond_epoch_and_closed_bar_time() -> None:
    open_time = 1_754_006_400_000_000
    close_time = 1_754_009_999_999_999
    frame = parse_spot_kline_zip(_archive(open_time, close_time), "BTCUSDT")
    assert frame.loc[0, "bar_open_time"] == pd.Timestamp("2025-08-01", tz="UTC")
    assert frame.loc[0, "feature_time"] == pd.Timestamp(
        "2025-08-01 01:00:00", tz="UTC"
    )
    assert frame.loc[0, "source_close_time"] < frame.loc[0, "feature_time"]
    assert frame.loc[0, "close"] == 101.0
