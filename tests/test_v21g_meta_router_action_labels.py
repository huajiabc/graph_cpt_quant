from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v21g_meta_router_action_labels import (
    V21GConfig,
    build_meta_router_labels,
)


def _row(
    idx: int,
    *,
    net: float,
    checkpoint_net: float,
    cp60: bool,
    protect: bool,
    late: bool,
) -> dict[str, object]:
    entry = pd.Timestamp("2025-08-01T00:00:00Z") + pd.Timedelta(hours=idx)
    return {
        "trade_key": f"trade-{idx}",
        "signal_id": f"trade-{idx}",
        "symbol": f"S{idx}USDT",
        "candidate": "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad",
        "entry_time": entry,
        "period": "search",
        "state_cluster_id": "SDG00",
        "cic_type": "CIC1" if idx % 2 == 0 else "CIC2",
        "btc_state": "BTC_up",
        "market_impulse_density": 0.5,
        "beta_strength": 99.0,
        "burst_count_so_far": 10 if late else 2,
        "net20": net,
        "checkpoint_net_at_cost": checkpoint_net,
        "cp60_would_exit": cp60,
        "beta_high_protect_candidate": protect,
        "late_burst_o6_candidate": late,
    }


def test_meta_router_action_labels_are_multi_head() -> None:
    membership = pd.DataFrame(
        [
            _row(0, net=0.02, checkpoint_net=-0.01, cp60=True, protect=True, late=True),
            _row(1, net=-0.02, checkpoint_net=-0.005, cp60=True, protect=False, late=True),
            _row(2, net=0.0005, checkpoint_net=0.0, cp60=False, protect=False, late=False),
        ]
    )

    labels = build_meta_router_labels(
        membership,
        V21GConfig(label_margin=0.001, small_trade_margin=0.003),
    )

    first = labels[labels["trade_key"].eq("trade-0")].iloc[0]
    second = labels[labels["trade_key"].eq("trade-1")].iloc[0]
    third = labels[labels["trade_key"].eq("trade-2")].iloc[0]

    assert first["pre_entry_action_label"] == "core_trade"
    assert first["post_entry_checkpoint_label"] == "keep_better_false_exit_risk"
    assert first["protect_a_label"] == "protect_keep_better"
    assert first["capacity_overflow_label"] == "allow_overflow"

    assert second["pre_entry_action_label"] == "no_trade"
    assert second["post_entry_checkpoint_label"] == "cp60_exit_better"
    assert second["capacity_overflow_label"] == "deny_overflow"

    assert third["pre_entry_action_label"] == "small_or_neutral"
    assert third["post_entry_checkpoint_label"] == "not_cp60_eligible"
    assert third["capacity_overflow_label"] == "not_overflow_eligible"
