from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v09a import add_cluster_graph_features


def test_cluster_impulse_density_excludes_own_symbol() -> None:
    month = pd.Timestamp("2026-01-01T00:00:00Z")
    times = pd.date_range(month - pd.Timedelta(days=3), periods=300, freq="1h")
    rows = []
    for ts in times:
        rows.extend(
            [
                {
                    "exchange": "bybit",
                    "symbol": "AAAUSDT",
                    "bar_open_time": ts,
                    "feature_time": ts,
                    "ret_1h": 0.01,
                    "ret_4h": 0.02,
                    "ret_4h_percentile": 96,
                    "dynamic_all_rank": 1,
                    "bullish_volume_shock_state": True,
                    "bullish_volume_shock_event": True,
                    "market_volume_impulse_density_high": True,
                },
                {
                    "exchange": "bybit",
                    "symbol": "BBBUSDT",
                    "bar_open_time": ts,
                    "feature_time": ts,
                    "ret_1h": 0.01,
                    "ret_4h": 0.01,
                    "ret_4h_percentile": 50,
                    "dynamic_all_rank": 2,
                    "bullish_volume_shock_state": False,
                    "bullish_volume_shock_event": False,
                    "market_volume_impulse_density_high": True,
                },
            ]
        )
    hist = pd.DataFrame(rows)
    membership = pd.DataFrame(
        [
            {"month_start": month, "symbol": "AAAUSDT", "cluster_id": "2026-01:C01", "cluster_size": 2},
            {"month_start": month, "symbol": "BBBUSDT", "cluster_id": "2026-01:C01", "cluster_size": 2},
        ]
    )

    live = hist[hist["feature_time"].eq(times[-1])].copy()
    enriched = add_cluster_graph_features(live, membership)
    aaa = enriched[enriched["symbol"].eq("AAAUSDT")].iloc[0]

    assert aaa["cluster_peer_count"] == 1
    assert aaa["cluster_impulse_density"] == 0.0
    assert not aaa["cluster_impulse_high"]
    assert aaa["c9a_cg4_market_high_cluster_low"]


def test_cluster_gate_turns_on_when_peer_impulses() -> None:
    month = pd.Timestamp("2026-01-01T00:00:00Z")
    membership = pd.DataFrame(
        [
            {"month_start": month, "symbol": "AAAUSDT", "cluster_id": "2026-01:C01", "cluster_size": 2},
            {"month_start": month, "symbol": "BBBUSDT", "cluster_id": "2026-01:C01", "cluster_size": 2},
        ]
    )
    live = pd.DataFrame(
        [
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "bar_open_time": month,
                "feature_time": month,
                "ret_4h": 0.02,
                "ret_4h_percentile": 96,
                "dynamic_all_rank": 1,
                "bullish_volume_shock_state": True,
                "bullish_volume_shock_event": True,
                "market_volume_impulse_density_high": True,
            },
            {
                "exchange": "bybit",
                "symbol": "BBBUSDT",
                "bar_open_time": month,
                "feature_time": month,
                "ret_4h": 0.01,
                "ret_4h_percentile": 80,
                "dynamic_all_rank": 2,
                "bullish_volume_shock_state": True,
                "bullish_volume_shock_event": True,
                "market_volume_impulse_density_high": True,
            },
        ]
    )

    enriched = add_cluster_graph_features(live, membership)
    aaa = enriched[enriched["symbol"].eq("AAAUSDT")].iloc[0]

    assert aaa["cluster_impulse_density"] == 1.0
    assert aaa["c9a_cg2_cic1_cluster_high_market_high"]
