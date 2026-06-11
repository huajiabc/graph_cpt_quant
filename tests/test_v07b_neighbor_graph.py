from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v07b import add_neighbor_graph_features, build_neighbor_graph_edges


def _toy_neighbor_rows() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01 00:00:00Z")
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    for idx in range(96 * 35):
        ts = start + pd.Timedelta(minutes=15 * idx)
        month_start = pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")
        if ts >= pd.Timestamp("2026-02-01 00:00:00Z"):
            month_start = pd.Timestamp("2026-02-01 00:00:00Z")
        for symbol in symbols:
            base_ret = 0.01 if idx % 20 < 10 else -0.01
            ret = base_ret if symbol in {"AAAUSDT", "BBBUSDT"} else -base_ret
            impulse = symbol in {"AAAUSDT", "BBBUSDT"} and idx % 96 == 0
            rows.append(
                {
                    "exchange": "bybit",
                    "symbol": symbol,
                    "bar_open_time": ts,
                    "bar_close_time": ts + pd.Timedelta(minutes=15),
                    "feature_time": ts + pd.Timedelta(minutes=15),
                    "month_start": month_start,
                    "dynamic_all_rank": {"AAAUSDT": 1, "BBBUSDT": 2, "CCCUSDT": 3}[symbol],
                    "ret_1h": ret,
                    "ret_4h": ret,
                    "ret_4h_percentile": 50,
                    "bullish_volume_shock_state": impulse,
                    "bullish_volume_shock_event": impulse,
                    "market_volume_impulse_density_high": True,
                    "close": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "open": 100.0,
                }
            )
    data = pd.DataFrame(rows)
    # Make the February AAA signal see BBB firing as a neighbor at the same feature time.
    signal_ts = pd.Timestamp("2026-02-01 00:15:00Z")
    mask = data["feature_time"].eq(signal_ts) & data["symbol"].isin(["AAAUSDT", "BBBUSDT"])
    data.loc[mask, "bullish_volume_shock_state"] = True
    data.loc[mask, "bullish_volume_shock_event"] = True
    return data


def test_v07b_neighbor_edges_and_gate_features_are_asof_monthly() -> None:
    data = _toy_neighbor_rows()
    edges = build_neighbor_graph_edges(data)

    feb_edges = edges[edges["month_start"].eq(pd.Timestamp("2026-02-01 00:00:00Z"))]
    assert not feb_edges.empty
    assert "return_corr_30d" in set(feb_edges["edge_type"])
    assert "volume_impulse_cooccurrence_30d" in set(feb_edges["edge_type"])

    featured = add_neighbor_graph_features(data, edges)
    signal = featured[
        featured["feature_time"].eq(pd.Timestamp("2026-02-01 00:15:00Z"))
        & featured["symbol"].eq("AAAUSDT")
    ].iloc[0]
    assert signal["neighbor_impulse_ratio"] > 0
    assert signal["gate_neighbor_impulse_high"]
    assert not signal["gate_isolated_signal"]
