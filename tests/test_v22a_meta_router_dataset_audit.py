from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v22a_meta_router_dataset_audit import (
    V22AConfig,
    write_v22a_meta_router_dataset_audit,
)


def test_meta_router_dataset_audit_blocks_labels_and_marks_preentry(tmp_path: Path) -> None:
    v21g = tmp_path / "v21g"
    v21g.mkdir()
    rows = []
    for idx in range(120):
        label = "core_trade" if idx < 70 else "no_trade"
        rows.append(
            {
                "trade_key": f"t{idx}",
                "symbol": "AAAUSDT" if idx % 2 else "BBBUSDT",
                "candidate": "CIC1_beta_extreme" if idx % 3 else "CIC2_beta_broad",
                "entry_time": f"2025-08-{1 + idx % 20:02d} 00:00:00+00:00",
                "entry_month": "2025-08" if idx < 80 else "2025-09",
                "period": "search" if idx < 90 else "validation",
                "state_cluster_id": "SDG00" if idx % 2 else "SDG01",
                "cic_type": "CIC1" if idx % 3 else "CIC2",
                "btc_state": "BTC_up",
                "market_impulse_density": 0.2 + idx / 1000,
                "cluster_density": 0.1,
                "beta_strength": 90 + idx % 10,
                "local_shock_strength": 3.0,
                "burst_count_so_far": idx % 12,
                "same_timestamp_peer_count": idx % 4,
                "walkforward_state_novelty": 0.5,
                "novelty_bucket": "q2",
                "net20": 0.01 if label == "core_trade" else -0.01,
                "pre_entry_action_label": label,
                "post_entry_checkpoint_label": "not_cp60_eligible",
                "protect_a_label": "not_protect_eligible",
                "capacity_overflow_label": "not_overflow_eligible",
                "utility_core_trade": 0.01,
                "utility_no_trade": 0.0,
                "utility_reduce_size_50": 0.005,
                "checkpoint_exit_delta_vs_keep": 0.0,
                "protect_keep_delta_vs_cp60": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(v21g / "meta_router_feature_matrix.csv", index=False)
    pd.DataFrame(
        [
            {"column": "market_impulse_density", "role": "feature"},
            {"column": "state_cluster_id", "role": "feature"},
            {"column": "pre_entry_action_label", "role": "label"},
            {"column": "net20", "role": "counterfactual_utility"},
            {"column": "utility_core_trade", "role": "counterfactual_utility"},
        ]
    ).to_csv(v21g / "router_dataset_schema.csv", index=False)

    outputs = write_v22a_meta_router_dataset_audit(
        V22AConfig(report_root=tmp_path / "out", v21g_root=v21g)
    )
    trainability = pd.read_csv(outputs["router_head_trainability"])
    leakage = pd.read_csv(outputs["feature_leakage_audit"])
    distribution = pd.read_csv(outputs["label_distribution"])

    pre = trainability[trainability["label_head"].eq("pre_entry_action_label")].iloc[0]
    assert pre["trainability_status"] == "trainable_first_pass_binary_core_vs_no_trade"
    assert distribution[distribution["label"].eq("core_trade")]["events"].iloc[0] == 70

    status_by_col = dict(zip(leakage["column"], leakage["audit_status"], strict=False))
    assert status_by_col["pre_entry_action_label"] == "blocked_label"
    assert status_by_col["net20"] == "blocked_counterfactual_or_realized_outcome"
    assert status_by_col["utility_core_trade"] == "blocked_counterfactual_or_realized_outcome"
    assert status_by_col["state_cluster_id"] == "ablation_only_high_overfit_risk"
    assert outputs["candidate_notes"].exists()
