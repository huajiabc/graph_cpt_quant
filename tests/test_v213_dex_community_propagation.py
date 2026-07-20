import numpy as np
import pandas as pd

from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    ALL_PEERS,
)
from pressure_graph.reports.v213_dex_community_propagation import (
    V213Config,
    beta_neutral_directional_weights,
    build_v213_events,
)


def test_directional_weights_are_unit_gross_and_beta_neutral() -> None:
    beta = pd.Series({"A": 1.2, "B": 0.8})
    weights = beta_neutral_directional_weights(["A", "B"], beta, -1.0)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    residual = weights["BTCUSDT"] + weights["A"] * 1.2 + weights["B"] * 0.8
    assert np.isclose(residual, 0.0)
    assert weights["A"] < 0 and weights["B"] < 0 and weights["BTCUSDT"] > 0


def test_event_builder_waits_and_uses_frozen_direction() -> None:
    feature_time = pd.Timestamp("2025-08-01 01:15", tz="UTC")
    entry_time = feature_time + pd.Timedelta(minutes=15)
    exit_time = entry_time + pd.Timedelta(hours=1)
    close = pd.DataFrame(
        {
            "BTCUSDT": [100.0, 101.0],
            "SRC": [100.0, 105.0],
            "P1": [100.0, 103.0],
            "P2": [100.0, 103.0],
            "P3": [100.0, 103.0],
            "P4": [100.0, 103.0],
        },
        index=[entry_time, exit_time],
    )
    candidate = pd.DataFrame(
        [
            {
                "candidate": ALL_PEERS,
                "event_id": "e1",
                "source_symbol": "SRC",
                "source_vendor": "dexpaprika_pool_ohlcv_1h",
                "chain": "eth",
                "event_time": feature_time - pd.Timedelta(minutes=75),
                "event_available_time": feature_time - pd.Timedelta(minutes=10),
                "feature_time": feature_time,
                "entry_time": entry_time,
                "period": "development",
                "entry_month": "2025-08",
                "community_id": "C1",
                "source_return_z": 2.0,
                "source_direction": 1,
                "selection_count": 4,
                "selection_symbols": "P1|P2|P3|P4",
            }
        ]
    )
    event_features = pd.DataFrame([{"event_id": "e1", "community_peer_symbols": "P1|P2|P3|P4"}])
    membership = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2025-08-01", tz="UTC")] * 5,
            "community_id": ["C1"] * 5,
            "symbol": ["SRC", "P1", "P2", "P3", "P4"],
        }
    )
    risk = pd.DataFrame(
        {
            "risk_month": [pd.Timestamp("2025-08-01", tz="UTC")] * 5,
            "receiver": ["SRC", "P1", "P2", "P3", "P4"],
            "btc_beta": [1.0] * 5,
        }
    )
    cfg = V213Config(holding_bars=4, risk_min_samples=1)
    events = build_v213_events(candidate, event_features, membership, risk, close, cfg)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["entry_time"] == entry_time
    assert row["exit_time"] == exit_time
    assert np.isclose(row["gross_notional"], 1.0)
    assert np.isclose(row["residual_btc_beta"], 0.0)
    assert row["gross_return"] > 0
