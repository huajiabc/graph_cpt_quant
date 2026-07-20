from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.cross_venue_tape_recorder import BINANCE_WS
from pressure_graph.cross_venue_tape import (
    MinuteBarAccumulator,
    combine_bar_fragments,
    coverage_summary,
    parse_binance_message,
    parse_bybit_message,
    read_bar_fragments,
    write_bar_fragment,
)


def test_binance_aggregate_trade_uses_market_route() -> None:
    assert "/market/stream?streams=" in BINANCE_WS


def test_parsers_preserve_aggressor_side() -> None:
    binance = parse_binance_message(
        json.dumps(
            {
                "stream": "aaausdt@aggTrade",
                "data": {
                    "e": "aggTrade",
                    "s": "AAAUSDT",
                    "a": 7,
                    "p": "100",
                    "q": "2",
                    "T": 1_767_225_605_000,
                    "m": True,
                },
            }
        )
    )
    assert binance[0]["side"] == "Sell"
    assert binance[0]["turnover"] == 200.0

    bybit = parse_bybit_message(
        {
            "topic": "publicTrade.AAAUSDT",
            "data": [
                {
                    "s": "AAAUSDT",
                    "i": "x",
                    "p": "101",
                    "v": "3",
                    "T": 1_767_225_606_000,
                    "S": "Buy",
                }
            ],
        }
    )
    assert bybit[0]["side"] == "Buy"
    assert bybit[0]["turnover"] == 303.0


def test_accumulator_and_fragment_merge_are_restart_safe() -> None:
    accumulator = MinuteBarAccumulator()
    base = {
        "exchange": "binance",
        "symbol": "AAAUSDT",
        "price": 100.0,
        "size": 1.0,
        "turnover": 100.0,
        "side": "Buy",
    }
    accumulator.add_trade(
        {**base, "timestamp": pd.Timestamp("2026-01-01 00:00:05Z"), "event_id": "1"},
        "session-a",
    )
    first = accumulator.drain(pd.Timestamp("2026-01-01 00:01:00Z"))

    accumulator.add_trade(
        {
            **base,
            "timestamp": pd.Timestamp("2026-01-01 00:00:40Z"),
            "event_id": "2",
            "price": 102.0,
            "turnover": 102.0,
            "side": "Sell",
        },
        "session-b",
    )
    second = accumulator.drain(pd.Timestamp("2026-01-01 00:01:00Z"))
    merged = combine_bar_fragments(pd.concat([first, second], ignore_index=True))

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["fragment_count"] == 2
    assert row["open"] == 100.0
    assert row["close"] == 102.0
    assert row["trade_count"] == 2
    assert row["buy_sell_imbalance"] == pytest.approx(-2.0 / 202.0)


def test_event_lag_measures_receipt_not_delayed_flush() -> None:
    accumulator = MinuteBarAccumulator()
    accumulator.add_trade(
        {
            "exchange": "binance",
            "symbol": "AAAUSDT",
            "timestamp": pd.Timestamp("2026-01-01 00:00:05Z"),
            "event_id": "1",
            "price": 100.0,
            "size": 1.0,
            "turnover": 100.0,
            "side": "Buy",
        },
        "session-a",
        received_at=pd.Timestamp("2026-01-01 00:00:07Z"),
    )
    bars = accumulator.drain(pd.Timestamp("2026-01-01 01:00:00Z"))

    assert bars.iloc[0]["event_lag_seconds"] == 2.0


def test_coverage_requires_synchronized_minutes() -> None:
    times = pd.date_range("2026-01-01 00:00Z", periods=3, freq="1min")
    rows = []
    for exchange in ("binance", "bybit"):
        for timestamp in times:
            if exchange == "bybit" and timestamp == times[1]:
                continue
            rows.append(
                {
                    "exchange": exchange,
                    "symbol": "AAAUSDT",
                    "bar_open_time": timestamp,
                    "fragment_count": 1,
                    "event_lag_seconds": 1.0,
                }
            )
    coverage = coverage_summary(
        pd.DataFrame(rows),
        pd.Timestamp("2026-01-01 00:00Z"),
        pd.Timestamp("2026-01-01 00:03Z"),
    )
    binance = coverage[coverage["exchange"].eq("binance")].iloc[0]
    bybit = coverage[coverage["exchange"].eq("bybit")].iloc[0]
    assert binance["coverage_ratio"] == 1.0
    assert bybit["coverage_ratio"] == pytest.approx(2 / 3)
    assert binance["synchronized_ratio"] == pytest.approx(2 / 3)


def test_fragment_reader_prunes_days_before_loading(tmp_path) -> None:
    old = pd.DataFrame(
        {
            "exchange": ["binance"],
            "symbol": ["AAAUSDT"],
            "bar_open_time": [pd.Timestamp("2026-01-01 00:00Z")],
        }
    )
    current = old.assign(bar_open_time=pd.Timestamp("2026-01-03 00:00Z"))
    write_bar_fragment(old, tmp_path, "old")
    write_bar_fragment(current, tmp_path, "current")

    loaded = read_bar_fragments(
        tmp_path,
        start=pd.Timestamp("2026-01-03 00:00Z"),
        end=pd.Timestamp("2026-01-04 00:00Z"),
    )
    assert loaded["bar_open_time"].tolist() == [pd.Timestamp("2026-01-03 00:00Z")]
