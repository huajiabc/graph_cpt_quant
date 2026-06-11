from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.orderbook import (
    compute_orderbook_features,
    normalize_orderbook_snapshot,
    select_orderbook_symbols,
)


def test_normalize_orderbook_snapshot_computes_depth_features() -> None:
    payload = {
        "s": "AAAUSDT",
        "ts": "1781154562000",
        "u": 123,
        "seq": 456,
        "b": [["99.9", "10"], ["99.8", "20"], ["99.0", "100"]],
        "a": [["100.1", "5"], ["100.2", "15"], ["101.0", "100"]],
    }

    levels, features = normalize_orderbook_snapshot(
        "AAAUSDT", payload, received_at=pd.Timestamp("2026-06-11 00:00:00Z")
    )

    assert len(levels) == 6
    row = features.iloc[0]
    assert row["best_bid"] == pytest.approx(99.9)
    assert row["best_ask"] == pytest.approx(100.1)
    assert row["spread_bps"] == pytest.approx(20.0)
    assert row["bid_depth_20bp"] == pytest.approx(99.9 * 10 + 99.8 * 20)
    assert row["ask_depth_20bp"] == pytest.approx(100.1 * 5 + 100.2 * 15)
    assert row["top5_bid_notional"] == pytest.approx(99.9 * 10 + 99.8 * 20 + 99.0 * 100)


def test_compute_orderbook_features_returns_empty_without_both_sides() -> None:
    levels = pd.DataFrame(
        [
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "snapshot_time": pd.Timestamp("2026-06-11 00:00:00Z"),
                "exchange_ts": pd.Timestamp("2026-06-11 00:00:00Z"),
                "update_id": 1,
                "seq": 1,
                "side": "bid",
                "level": 1,
                "price": 100.0,
                "size": 1.0,
                "notional": 100.0,
            }
        ]
    )

    assert compute_orderbook_features(levels).empty


def test_select_orderbook_symbols_prefers_demand_queue(tmp_path) -> None:
    demand = pd.DataFrame(
        [
            {
                "symbol": "CICUSDT",
                "priority": "P2",
                "status": "pending",
                "window_end": pd.Timestamp("2026-06-11 00:00:00Z"),
            }
        ]
    )
    demand_path = tmp_path / "demand.parquet"
    demand.to_parquet(demand_path, index=False)
    features = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "feature_time": pd.Timestamp("2026-06-11 00:00:00Z"),
                "dynamic_all_rank": 1,
            }
        ]
    )
    feature_path = tmp_path / "features.parquet"
    features.to_parquet(feature_path, index=False)

    selected = select_orderbook_symbols(
        demand_path,
        feature_path,
        max_symbols=2,
        core_reference_symbols=("ETHUSDT",),
    )

    assert selected == ["CICUSDT", "ETHUSDT"]
