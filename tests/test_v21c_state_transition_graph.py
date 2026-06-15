from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v21c_state_transition_graph import (
    _build_edges,
    _edge_summary,
    _holdout_autopsy,
    _path_summary,
)


def _row(idx: int, *, symbol: str, cluster: str, period: str, net: float, burst: str) -> dict[str, object]:
    entry = pd.Timestamp("2026-05-01T00:00:00Z") + pd.Timedelta(minutes=15 * idx)
    return {
        "signal_id": f"sig-{idx}",
        "trade_key": f"sig-{idx}",
        "symbol": symbol,
        "candidate": "CIC1_beta_extreme",
        "cic_type": "CIC1",
        "state_cluster_id": cluster,
        "period": period,
        "entry_time": entry,
        "net20": net,
        "mfe_12h": max(net, 0.0),
        "mae_12h": min(net, 0.0),
        "hit_10pct_12h": net > 0.10,
        "burst_id": burst,
    }


def test_state_transition_edges_and_summaries() -> None:
    membership = pd.DataFrame(
        [
            _row(0, symbol="AAAUSDT", cluster="SDG00", period="search", net=0.03, burst="b1"),
            _row(1, symbol="BBBUSDT", cluster="SDG01", period="search", net=0.02, burst="b1"),
            _row(2, symbol="AAAUSDT", cluster="SDG02", period="holdout", net=-0.04, burst="b2"),
            _row(3, symbol="BBBUSDT", cluster="SDG02", period="holdout", net=-0.01, burst="b2"),
            _row(4, symbol="AAAUSDT", cluster="SDG03", period="holdout", net=0.05, burst="b3"),
        ]
    )

    edges = _build_edges(membership)
    assert not edges.empty
    assert {"same_symbol_event", "same_burst_event", "global_time_event"}.issubset(set(edges["relation"]))

    summary = _edge_summary(edges)
    assert not summary.empty
    assert {"transition_count", "transition_probability_from_source", "holdout_target_net20"}.issubset(summary.columns)

    paths = _path_summary(edges)
    assert "path_count" in paths.columns or paths.empty

    holdout = _holdout_autopsy(edges)
    assert not holdout.empty
    assert holdout.iloc[0]["target_period"] == "holdout"
