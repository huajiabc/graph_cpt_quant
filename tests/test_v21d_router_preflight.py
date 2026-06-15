from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v21d_router_preflight import (
    V21DConfig,
    _add_transition_keys,
    _build_rules,
    _cluster_stats,
    _evaluate_rules,
    _transition_stats,
)


def _row(idx: int, *, cluster: str, period: str, net: float) -> dict[str, object]:
    entry = pd.Timestamp("2025-10-01T00:00:00Z") + pd.Timedelta(minutes=30 * idx)
    return {
        "symbol": f"S{idx % 3}USDT",
        "candidate": "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad",
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=2),
        "checkpoint_time": entry + pd.Timedelta(hours=1),
        "checkpoint_price_covered": True,
        "checkpoint_net_at_cost": min(net, -0.001),
        "net_return_at_cost": net,
        "net20": net,
        "effective_net_return": net,
        "weighted_return": net,
        "month": entry.strftime("%Y-%m"),
        "period": period,
        "state_cluster_id": cluster,
        "burst_id": f"b{idx // 3}",
        "burst_count_so_far": idx + 1,
        "signal_id": f"sig-{idx}",
        "trade_key": f"sig-{idx}",
    }


def test_router_preflight_builds_safe_and_diagnostic_rules() -> None:
    rows = []
    for idx in range(6):
        rows.append(_row(idx, cluster="BAD", period="search", net=-0.02))
    for idx in range(6, 12):
        rows.append(_row(idx, cluster="FLIP", period="search", net=0.03))
    rows.append(_row(12, cluster="FLIP", period="holdout", net=-0.04))
    rows.append(_row(13, cluster="BAD", period="holdout", net=-0.01))
    membership = _add_transition_keys(pd.DataFrame(rows))

    clusters = _cluster_stats(membership)
    transitions = _transition_stats(membership)
    rules = _build_rules(clusters, transitions, V21DConfig(min_cluster_events=3, min_transition_events=2))

    assert rules["rule_id"].eq("baseline_B4").any()
    assert rules["leakage_status"].eq("safe").any()
    assert rules["leakage_status"].eq("holdout_leakage_diagnostic").any()

    summary, ledger = _evaluate_rules(membership, rules, V21DConfig(min_cluster_events=3, min_transition_events=2))
    assert not summary.empty
    assert "holdout_delta_vs_baseline_net20" in summary.columns
    assert not ledger.empty
