from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v21f_state_drift_audit import (
    V21FConfig,
    _cluster_period_drift,
    _feature_period_drift,
    _walkforward_novelty,
)


FEATURES = ["market_impulse_density", "beta_strength", "burst_count_so_far"]


def _row(idx: int, *, month: str, period: str, cluster: str, beta: float, net: float) -> dict[str, object]:
    entry = pd.Timestamp(f"{month}-01T00:00:00Z") + pd.Timedelta(hours=idx)
    return {
        "trade_key": f"{month}-{idx}",
        "signal_id": f"{month}-{idx}",
        "symbol": f"S{idx % 2}USDT",
        "candidate": "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad",
        "entry_time": entry,
        "period": period,
        "state_cluster_id": cluster,
        "cic_type": "CIC1" if idx % 2 == 0 else "CIC2",
        "btc_state": "BTC_up",
        "market_impulse_density": 0.7,
        "beta_strength": beta,
        "burst_count_so_far": idx + 1,
        "net20": net,
    }


def test_state_drift_tables_and_walkforward_novelty() -> None:
    rows = []
    for idx in range(5):
        rows.append(_row(idx, month="2025-07", period="search", cluster="SDG00", beta=1.0 + idx * 0.1, net=0.01))
    for idx in range(5, 10):
        rows.append(_row(idx, month="2025-08", period="validation", cluster="SDG00", beta=1.2 + idx * 0.1, net=0.02))
    for idx in range(10, 13):
        rows.append(_row(idx, month="2026-05", period="holdout", cluster="SDG00", beta=5.0, net=-0.03))
    membership = pd.DataFrame(rows)
    cfg = V21FConfig(min_cluster_reference_events=3, min_history_months=1)

    feature_drift = _feature_period_drift(membership, FEATURES)
    cluster_drift = _cluster_period_drift(membership, FEATURES, cfg)
    novelty = _walkforward_novelty(membership, FEATURES, cfg)

    assert feature_drift["period"].eq("holdout").any()
    holdout_cluster = cluster_drift[
        cluster_drift["period"].eq("holdout") & cluster_drift["state_cluster_id"].eq("SDG00")
    ].iloc[0]
    assert holdout_cluster["cluster_feature_drift_score"] > 0
    assert "beta_strength" in str(holdout_cluster["dominant_drift_features"])

    first_month = novelty[novelty["entry_month"].eq("2025-07")]
    later_months = novelty[novelty["entry_month"].ne("2025-07")]
    assert first_month["novelty_status"].eq("insufficient_history").all()
    assert later_months["novelty_status"].eq("covered").any()
