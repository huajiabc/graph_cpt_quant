from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v21e_walkforward_router_stability import (
    V21EConfig,
    _target_join,
    _target_split,
    evaluate_walkforward_router,
)


def _row(idx: int, *, month: str, cluster: str, net: float) -> dict[str, object]:
    entry = pd.Timestamp(f"{month}-01T00:00:00Z") + pd.Timedelta(hours=3 * idx)
    return {
        "symbol": f"S{idx % 4}USDT",
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
        "period": "search",
        "state_cluster_id": cluster,
        "burst_id": f"{month}-b{idx // 3}",
        "burst_count_so_far": idx + 1,
        "signal_id": f"{month}-sig-{idx}",
        "trade_key": f"{month}-sig-{idx}",
    }


def test_walkforward_router_uses_prior_months_only() -> None:
    rows = []
    for idx in range(3):
        rows.append(_row(idx, month="2025-07", cluster="BAD", net=-0.02))
    for idx in range(3, 6):
        rows.append(_row(idx, month="2025-08", cluster="BAD", net=0.03))
    for idx in range(6, 9):
        rows.append(_row(idx, month="2025-08", cluster="GOOD", net=0.01))

    results = evaluate_walkforward_router(
        pd.DataFrame(rows),
        V21EConfig(min_history_months=1, min_cluster_events=2, min_transition_events=99),
    )

    monthly = results["monthly"]
    assert not monthly.empty
    first_month_rule = monthly[
        monthly["eval_month"].eq("2025-07")
        & monthly["rule_id"].eq("no_trade_prior_bad_clusters")
    ].iloc[0]
    second_month_rule = monthly[
        monthly["eval_month"].eq("2025-08")
        & monthly["rule_id"].eq("no_trade_prior_bad_clusters")
    ].iloc[0]

    assert int(first_month_rule["target_count"]) == 0
    assert int(second_month_rule["target_count"]) == 1
    assert int(second_month_rule["router_affected_events"]) == 3

    summary = results["summary"]
    assert "validation_delta_vs_baseline_net20" in summary.columns
    assert summary["rule_id"].eq("baseline_B4").any()


def test_transition_target_separator_preserves_pipe_keys() -> None:
    key = "global_time_event|h1|SDG01->SDG02"
    joined = _target_join([key])
    assert _target_split(joined) == [key]
