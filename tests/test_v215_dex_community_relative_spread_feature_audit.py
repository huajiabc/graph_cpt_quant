import pandas as pd

from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    ALL_PEERS,
)
from pressure_graph.reports.v215_dex_community_relative_spread_feature_audit import (
    build_v215_spread_features,
)


def test_balanced_peer_ranking_uses_feature_close_only() -> None:
    index = pd.date_range("2025-08-01 00:15", periods=6, freq="15min", tz="UTC")
    close = pd.DataFrame(
        {
            "P1": [100, 100, 100, 100, 99, 99],
            "P2": [100, 100, 100, 100, 100, 100],
            "P3": [100, 100, 100, 100, 101, 101],
            "P4": [100, 100, 100, 100, 102, 102],
            "P5": [100, 100, 100, 100, 103, 103],
        },
        index=index,
    )
    feature_time = index[-2]
    features = pd.DataFrame(
        [
            {
                ALL_PEERS: True,
                "event_id": "e1",
                "source_symbol": "SRC",
                "source_vendor": "dex",
                "chain": "eth",
                "event_time": index[0],
                "event_available_time": feature_time - pd.Timedelta(minutes=10),
                "feature_time": feature_time,
                "entry_time": index[-1],
                "period": "development",
                "entry_month": "2025-08",
                "community_id": "C1",
                "source_return_z": 2.0,
                "source_direction": 1,
                "community_peer_symbols": "P1|P2|P3|P4|P5",
            }
        ]
    )
    candidate = build_v215_spread_features(features, close)
    assert len(candidate) == 1
    row = candidate.iloc[0]
    assert row["leg_size"] == 2
    assert set(row["laggard_symbols"].split("|")) == {"P1", "P2"}
    assert set(row["leader_symbols"].split("|")) == {"P4", "P5"}
    assert row["ranking_window_end"] == feature_time
