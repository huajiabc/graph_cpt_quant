from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from pressure_graph.orderflow import summarize_trade_window
from pressure_graph.orderflow_history import (
    EVENT_WINDOWS,
    PRE_ENTRY_WINDOWS,
    OrderflowHistoryConfig,
    binance_symbol_candidates,
    build_event_orderflow,
    event_windows,
    extract_events,
    normalize_aggtrades,
    signal_event_id,
)
from pressure_graph.reports.v07d1 import _signal_id


def _aggtrades_frame(rows: list[tuple[int, float, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "agg_trade_id": range(1, len(rows) + 1),
            "price": [row[1] for row in rows],
            "quantity": [row[2] for row in rows],
            "first_trade_id": range(1, len(rows) + 1),
            "last_trade_id": range(1, len(rows) + 1),
            "transact_time": [row[0] for row in rows],
            "is_buyer_maker": [row[3] for row in rows],
        }
    )


def test_binance_symbol_candidates_alias_first():
    candidates = binance_symbol_candidates("SHIB1000USDT")
    assert candidates[0] == "1000SHIBUSDT"
    assert "SHIB1000USDT" in candidates


def test_binance_symbol_candidates_generic_multiplier_forms():
    candidates = binance_symbol_candidates("10000SATSUSDT")
    assert "SATSUSDT" in candidates
    assert "1000SATSUSDT" in candidates
    assert candidates[0] == "10000SATSUSDT"


def test_normalize_aggtrades_buyer_maker_is_taker_sell():
    base_ms = 1_750_000_000_000
    frame = _aggtrades_frame(
        [
            (base_ms, 100.0, 2.0, False),
            (base_ms + 1_000, 101.0, 3.0, True),
        ]
    )
    normalized = normalize_aggtrades(frame, "TESTUSDT")
    assert list(normalized["side"]) == ["buy", "sell"]
    assert float(normalized["turnover"].iloc[0]) == 200.0
    assert normalized["timestamp"].dt.tz is not None


def test_normalize_aggtrades_drops_header_row_and_reads_microseconds():
    base_us = 1_750_000_000_000_000
    frame = _aggtrades_frame([(base_us, 10.0, 1.0, False), (base_us + 1_000_000, 11.0, 1.0, True)])
    header = pd.DataFrame(
        [
            {
                "agg_trade_id": "agg_trade_id",
                "price": "price",
                "quantity": "quantity",
                "first_trade_id": "first_trade_id",
                "last_trade_id": "last_trade_id",
                "transact_time": "transact_time",
                "is_buyer_maker": "is_buyer_maker",
            }
        ]
    )
    normalized = normalize_aggtrades(pd.concat([header, frame], ignore_index=True), "TESTUSDT")
    assert len(normalized) == 2
    assert normalized["timestamp"].iloc[0].year == 2025


def test_event_windows_pre_entry_contract():
    signal_time = pd.Timestamp("2026-01-02 03:15:00", tz="UTC")
    entry_time = pd.Timestamp("2026-01-02 04:00:00", tz="UTC")
    windows = event_windows(signal_time, entry_time)
    assert set(windows) == set(EVENT_WINDOWS)
    assert windows["shock_bar"] == (signal_time - pd.Timedelta(minutes=15), signal_time)
    assert windows["reclaim_bar"] == (entry_time - pd.Timedelta(minutes=15), entry_time)
    assert windows["pullback_window"][1] <= entry_time - pd.Timedelta(minutes=15)
    for name in PRE_ENTRY_WINDOWS:
        assert windows[name][1] <= entry_time
    assert windows["entry_bar"][0] == entry_time


def test_event_windows_immediate_entry_collapses_pullback():
    signal_time = pd.Timestamp("2026-01-02 03:15:00", tz="UTC")
    entry_time = signal_time + pd.Timedelta(minutes=15)
    windows = event_windows(signal_time, entry_time)
    start, end = windows["pullback_window"]
    assert end <= start


def test_extract_events_dedupes_cost_frames_and_matches_v07d1_signal_id():
    trades = pd.DataFrame(
        {
            "exchange": ["bybit"] * 4,
            "symbol": ["AAAUSDT"] * 4,
            "candidate": ["CIC1_beta_extreme"] * 4,
            "signal_time": [pd.Timestamp("2026-01-02 03:15:00", tz="UTC")] * 4,
            "entry_time": [pd.Timestamp("2026-01-02 04:00:00", tz="UTC")] * 4,
            "cost_single_side_bps": [5.0, 20.0, 30.0, 50.0],
        }
    )
    events = extract_events(trades, ("CIC1_beta_extreme",))
    assert len(events) == 1
    expected = _signal_id(trades.head(1)).iloc[0]
    assert events["signal_id"].iloc[0] == expected
    assert events["signal_id"].iloc[0] == signal_event_id(
        "bybit", "AAAUSDT", trades["signal_time"].iloc[0], "CIC1_beta_extreme"
    )


def test_summarize_trade_window_assume_normalized_parity():
    base = pd.Timestamp("2026-01-02 03:00:00", tz="UTC")
    rows = []
    for minute in range(30):
        rows.append(
            {
                "exchange": "binance",
                "symbol": "AAAUSDT",
                "timestamp": base + pd.Timedelta(minutes=minute),
                "execId": str(minute),
                "price": 100.0 + minute,
                "size": 1.0 + (minute % 3),
                "turnover": (100.0 + minute) * (1.0 + (minute % 3)),
                "side": "buy" if minute % 2 == 0 else "sell",
            }
        )
    trades = pd.DataFrame(rows)
    start = base + pd.Timedelta(minutes=5)
    end = base + pd.Timedelta(minutes=20)
    default_stats = summarize_trade_window(trades, start, end)
    normalized_stats = summarize_trade_window(trades, start, end, assume_normalized=True)
    for key, value in default_stats.items():
        other = normalized_stats[key]
        if isinstance(value, float) and np.isnan(value):
            assert isinstance(other, float) and np.isnan(other)
        else:
            assert other == value


def test_build_event_orderflow_offline(monkeypatch, tmp_path):
    signal_time = pd.Timestamp("2026-01-02 03:15:00", tz="UTC")
    entry_time = pd.Timestamp("2026-01-02 04:00:00", tz="UTC")
    events = pd.DataFrame(
        {
            "exchange": ["bybit"],
            "symbol": ["AAAUSDT"],
            "candidate": ["CIC1_beta_extreme"],
            "signal_time": [signal_time],
            "entry_time": [entry_time],
        }
    )
    events["signal_id"] = [
        signal_event_id("bybit", "AAAUSDT", signal_time, "CIC1_beta_extreme")
    ]

    day_start = pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    rows = []
    for minute in range(0, 24 * 60):
        ts = day_start + pd.Timedelta(minutes=minute)
        rows.append(
            {
                "exchange": "binance",
                "symbol": "AAAUSDT",
                "timestamp": ts,
                "execId": str(minute),
                "price": 100.0,
                "size": 1.0,
                "turnover": 100.0,
                "side": "buy" if minute % 4 else "sell",
            }
        )
    day_frame = pd.DataFrame(rows)

    fake_paths: dict[tuple[str, date], str] = {}

    def fake_download(binance_symbol, day, cfg):
        fake_paths[(binance_symbol, day)] = "ok"
        return tmp_path / f"{binance_symbol}-{day}.zip"

    def fake_load(path):
        return day_frame

    monkeypatch.setattr("pressure_graph.orderflow_history.download_aggtrades_day", fake_download)
    monkeypatch.setattr("pressure_graph.orderflow_history.load_aggtrades_zip", fake_load)

    cfg = OrderflowHistoryConfig(history_root=tmp_path / "history", download_workers=2)
    features = build_event_orderflow(events, cfg)
    assert len(features) == 1
    row = features.iloc[0]
    assert row["binance_symbol"] == "AAAUSDT"
    assert row["mapping_status"] == "mapped"
    assert bool(row["reclaim_bar_covered"])
    assert row["reclaim_bar_trade_count"] == 15
    assert 0.0 < row["reclaim_bar_taker_buy_ratio"] < 1.0
    assert bool(row["shock_bar_covered"])
    assert bool(row["post_entry_1h_covered"])
    for window in EVENT_WINDOWS:
        assert f"{window}_coverage_ratio" in features.columns
