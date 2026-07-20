from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd

from pressure_graph.binance_option_eoh_history import (
    OPTION_EOH_COLUMNS,
    parse_option_eoh_zip,
    parse_um_hourly_kline_zip,
)


def _zip_csv(name: str, csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, csv_text)
    return buffer.getvalue()


def test_parse_option_eoh_uses_conservative_hour_end() -> None:
    values = {
        "date": "2023-10-20",
        "hour": 0,
        "symbol": "BTC-231229-30000-C",
        "underlying": "BTCUSDT",
        "type": "C",
        "strike": "231229-30000",
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume_contracts": 1,
        "volume_usdt": 100,
        "best_bid_price": 99,
        "best_ask_price": 101,
        "best_bid_qty": 2,
        "best_ask_qty": 3,
        "best_buy_iv": 0.50,
        "best_sell_iv": 0.51,
        "mark_price": 100,
        "mark_iv": 0.505,
        "delta": 0.5,
        "gamma": 0.001,
        "vega": 20,
        "theta": -5,
        "openinterest_contracts": 10,
        "openinterest_usdt": 300_000,
    }
    csv = pd.DataFrame([[values[column] for column in OPTION_EOH_COLUMNS]], columns=OPTION_EOH_COLUMNS)
    parsed = parse_option_eoh_zip(
        _zip_csv("sample.csv", csv.to_csv(index=False)), date(2023, 10, 20)
    )
    assert len(parsed) == 1
    assert parsed.loc[0, "snapshot_time"] == pd.Timestamp("2023-10-20 01:00:00Z")
    assert parsed.loc[0, "expiration_time"] == pd.Timestamp("2023-12-29 08:00:00Z")
    assert parsed.loc[0, "strike_price"] == 30_000
    assert parsed.loc[0, "option_type"] == "C"


def test_parse_um_hourly_kline_zip() -> None:
    row = [
        1_697_766_000_000,
        28_000,
        28_100,
        27_900,
        28_050,
        100,
        1_697_769_599_999,
        2_805_000,
        500,
        55,
        1_542_750,
        0,
    ]
    parsed = parse_um_hourly_kline_zip(
        _zip_csv("BTCUSDT-1h.csv", ",".join(map(str, row))), "BTCUSDT"
    )
    assert len(parsed) == 1
    assert parsed.loc[0, "symbol"] == "BTCUSDT"
    assert parsed.loc[0, "close"] == 28_050
    assert parsed.loc[0, "bar_close_time"] > parsed.loc[0, "bar_open_time"]
