from __future__ import annotations

import pandas as pd

from pressure_graph.recent_perp_carry_history import parse_binance_um_klines


def test_parse_recent_binance_kline_sets_closed_feature_time() -> None:
    rows = [
        [
            1_754_006_400_000,
            "100",
            "102",
            "99",
            "101",
            "10",
            1_754_009_999_999,
            "1000",
            42,
            "4",
            "400",
            "0",
        ]
    ]
    frame = parse_binance_um_klines(rows, "BTCUSDT")
    assert frame.loc[0, "feature_time"] == pd.Timestamp(
        "2025-08-01 01:00:00", tz="UTC"
    )
    assert frame.loc[0, "close"] == 101.0
