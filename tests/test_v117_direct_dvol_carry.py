from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v117_direct_dvol_carry import build_v117_contract_trade


def test_contract_trade_uses_later_positive_volume_entry() -> None:
    expiry = pd.Timestamp("2025-01-29 08:00:00", tz="UTC")
    target = expiry - pd.Timedelta(days=14)
    futures = pd.DataFrame(
        {
            "instrument_name": ["BTCDVOL_USDC-29JAN25"] * 5,
            "bar_open_time": [
                target - pd.Timedelta(hours=1),
                target,
                target + pd.Timedelta(hours=1),
                expiry,
                expiry + pd.Timedelta(hours=1),
            ],
            "expiration_time": [expiry] * 5,
            "open": [55.0, 56.0, 57.0, 50.0, 50.0],
            "close": [56.0, 57.0, 58.0, 50.0, 50.0],
            "volume": [1.0, 1.0, 2.0, 0.0, 0.0],
        }
    )
    dvol = pd.DataFrame(
        {"dvol_time": [target], "close": [50.0]}
    )
    trade = build_v117_contract_trade(futures, dvol)
    assert trade is not None
    assert trade["entry_time"] == target + pd.Timedelta(hours=1)
    assert trade["entry_price"] == 57.0
    assert trade["direction"] == -1.0
    assert trade["gross_return"] > 0


def test_contract_trade_rejects_stale_dvol() -> None:
    expiry = pd.Timestamp("2025-01-29 08:00:00", tz="UTC")
    target = expiry - pd.Timedelta(days=14)
    futures = pd.DataFrame(
        {
            "instrument_name": ["X"] * 3,
            "bar_open_time": [target, target + pd.Timedelta(hours=1), expiry],
            "expiration_time": [expiry] * 3,
            "open": [55.0, 55.0, 50.0],
            "close": [55.0, 55.0, 50.0],
            "volume": [1.0, 1.0, 0.0],
        }
    )
    dvol = pd.DataFrame(
        {"dvol_time": [target - pd.Timedelta(hours=3)], "close": [50.0]}
    )
    assert build_v117_contract_trade(futures, dvol) is None
