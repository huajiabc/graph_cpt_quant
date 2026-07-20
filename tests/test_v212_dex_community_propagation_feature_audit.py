import numpy as np
import pandas as pd

from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    ALL_PEERS,
    LAGGARD_PEERS,
    V212FeatureConfig,
    build_v212_candidate_features,
    build_v212_event_features,
    build_v212_source_context,
)


def test_source_context_uses_strictly_prior_scale() -> None:
    index = pd.date_range("2025-07-01", periods=16, freq="15min", tz="UTC")
    close = pd.DataFrame({"A": np.linspace(100.0, 120.0, len(index))}, index=index)
    cfg = V212FeatureConfig(source_lookback_bars=4, source_minimum_bars=3)
    returns, mean, scale, zscore = build_v212_source_context(close, cfg)
    timestamp = index[-1]
    expected_history = returns["A"].iloc[-5:-1]
    assert np.isclose(mean.at[timestamp, "A"], expected_history.mean())
    assert np.isclose(scale.at[timestamp, "A"], expected_history.std(ddof=1))
    assert np.isclose(
        zscore.at[timestamp, "A"],
        (returns.at[timestamp, "A"] - expected_history.mean()) / expected_history.std(ddof=1),
    )


def test_event_feature_timing_and_peer_rules() -> None:
    index = pd.date_range("2025-07-01", "2025-08-01 01:30", freq="15min", tz="UTC")
    close = pd.DataFrame(100.0, index=index, columns=["SRC", "P1", "P2", "P3", "P4"])
    close["SRC"] = 100.0 + 0.1 * np.sin(np.arange(len(index)) / 4.0)
    close.loc[index[-2] :, "SRC"] = 120.0
    close.loc[index[-2] :, "P1"] = 99.0
    close.loc[index[-2] :, "P2"] = 99.0
    close.loc[index[-2] :, "P3"] = 99.0
    close.loc[index[-2] :, "P4"] = 101.0
    event_time = pd.Timestamp("2025-08-01 00:00", tz="UTC")
    available = pd.Timestamp("2025-08-01 01:05", tz="UTC")
    events = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "cex_symbol": "SRC",
                "chain": "eth",
                "source": "dexpaprika_pool_ohlcv_1h",
                "mapping_confidence": "B",
                "event_time": event_time,
                "event_available_time": available,
                "zscore": 3.0,
                "percentile": 0.99,
            }
        ]
    )
    membership = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2025-08-01", tz="UTC")] * 5,
            "community_id": ["C1"] * 5,
            "symbol": ["SRC", "P1", "P2", "P3", "P4"],
        }
    )
    cfg = V212FeatureConfig(
        source_lookback_bars=96,
        source_minimum_bars=48,
        source_z_threshold=1.0,
    )
    features = build_v212_event_features(events, close, membership, cfg)
    assert len(features) == 1
    row = features.iloc[0]
    assert row["feature_time"] == pd.Timestamp("2025-08-01 01:15", tz="UTC")
    assert row["entry_time"] == pd.Timestamp("2025-08-01 01:30", tz="UTC")
    assert row["source_scale_history_end"] < row["feature_time"]
    assert bool(row[ALL_PEERS])
    assert bool(row[LAGGARD_PEERS])
    candidates = build_v212_candidate_features(features)
    assert set(candidates["candidate"]) == {ALL_PEERS, LAGGARD_PEERS}
    assert candidates.set_index("candidate").at[ALL_PEERS, "selection_count"] == 4
    assert candidates.set_index("candidate").at[LAGGARD_PEERS, "selection_count"] == 3
