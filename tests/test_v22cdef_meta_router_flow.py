from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v22a_meta_router_dataset_audit import V22AConfig, write_v22a_meta_router_dataset_audit
from pressure_graph.reports.v22b_preentry_meta_router import V22BConfig, write_v22b_preentry_meta_router
from pressure_graph.reports.v22c_walkforward_policy_simulation import (
    V22CConfig,
    write_v22c_walkforward_policy_simulation,
)
from pressure_graph.reports.v22d_threshold_stability import V22DConfig, write_v22d_threshold_stability
from pressure_graph.reports.v22e_negative_controls import V22EConfig, write_v22e_negative_controls
from pressure_graph.reports.v22f_meta_router_synthesis import V22FConfig, write_v22f_meta_router_synthesis


def _row(idx: int, month: str, label: str) -> dict[str, object]:
    entry = pd.Timestamp(f"{month}-02T00:00:00Z") + pd.Timedelta(minutes=15 * idx)
    exit_time = entry + pd.Timedelta(hours=4)
    net = 0.018 if label == "core_trade" else -0.018
    return {
        "trade_key": f"t{idx}",
        "symbol": f"S{idx % 8}USDT",
        "candidate": "CIC1_beta_extreme" if idx % 2 else "CIC2_beta_broad",
        "entry_time": entry.isoformat(),
        "entry_month": month,
        "period": "search" if month < "2026-02" else "validation",
        "exit_time": exit_time.isoformat(),
        "checkpoint_time": (entry + pd.Timedelta(hours=1)).isoformat(),
        "checkpoint_price_covered": True,
        "checkpoint_net_at_cost": -0.006 if label == "no_trade" else 0.004,
        "net_return_at_cost": net,
        "net20": net,
        "pre_entry_action_label": label,
        "post_entry_checkpoint_label": "cp60_exit_better" if label == "no_trade" else "not_cp60_eligible",
        "protect_a_label": "not_protect_eligible",
        "capacity_overflow_label": "not_overflow_eligible",
        "cic_type": "CIC1" if idx % 2 else "CIC2",
        "btc_state": "BTC_up" if idx % 4 else "BTC_chop",
        "market_impulse_density": 0.65 if label == "core_trade" else 0.12,
        "cluster_density": 0.2,
        "beta_strength": 99.0 if label == "core_trade" else 85.0,
        "local_shock_strength": 4.0,
        "ret_4h": 0.04 if label == "core_trade" else 0.002,
        "ret_4h_percentile": 99.0 if label == "core_trade" else 80.0,
        "symbol_volatility_percentile": 70.0,
        "burst_count_so_far": idx % 12,
        "minutes_since_burst_start": float((idx % 4) * 15),
        "same_timestamp_peer_count": idx % 5,
        "burst_id": f"burst-{month}-{idx // 5}",
        "beta_extreme_strength_high": label == "core_trade",
    }


def _write_v21g(root: Path) -> None:
    root.mkdir()
    rows = []
    idx = 0
    for month in ["2025-07", "2025-08", "2025-09", "2025-10", "2026-02", "2026-03"]:
        for inner in range(24):
            label = "core_trade" if inner % 3 else "no_trade"
            rows.append(_row(idx, month, label))
            idx += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "meta_router_event_labels.csv", index=False)
    feature_cols = [
        "trade_key",
        "symbol",
        "candidate",
        "entry_time",
        "entry_month",
        "period",
        "cic_type",
        "btc_state",
        "market_impulse_density",
        "cluster_density",
        "beta_strength",
        "local_shock_strength",
        "ret_4h",
        "ret_4h_percentile",
        "symbol_volatility_percentile",
        "burst_count_so_far",
        "minutes_since_burst_start",
        "same_timestamp_peer_count",
        "net20",
        "pre_entry_action_label",
        "post_entry_checkpoint_label",
        "protect_a_label",
        "capacity_overflow_label",
    ]
    frame[feature_cols].to_csv(root / "meta_router_feature_matrix.csv", index=False)
    pd.DataFrame(
        [{"column": col, "role": "feature"} for col in feature_cols if col not in {"net20", "pre_entry_action_label"}]
        + [
            {"column": "net20", "role": "counterfactual_utility"},
            {"column": "pre_entry_action_label", "role": "label"},
        ]
    ).to_csv(root / "router_dataset_schema.csv", index=False)


def test_v22cdef_reports_build_from_v22b_predictions(tmp_path: Path) -> None:
    v21g = tmp_path / "v21g"
    _write_v21g(v21g)
    a_root = tmp_path / "a"
    b_root = tmp_path / "b"
    c_root = tmp_path / "c"
    d_root = tmp_path / "d"
    e_root = tmp_path / "e"
    f_root = tmp_path / "f"

    write_v22a_meta_router_dataset_audit(V22AConfig(report_root=a_root, v21g_root=v21g))
    write_v22b_preentry_meta_router(
        V22BConfig(
            report_root=b_root,
            v21g_root=v21g,
            min_train_months=2,
            min_train_events=40,
            random_permutations=5,
            logistic_steps=80,
        )
    )
    c_cfg = V22CConfig(report_root=c_root, v22b_root=b_root)
    c_outputs = write_v22c_walkforward_policy_simulation(c_cfg)
    d_outputs = write_v22d_threshold_stability(V22DConfig(report_root=d_root, v22c=c_cfg))
    e_outputs = write_v22e_negative_controls(V22EConfig(report_root=e_root, v22c=c_cfg, permutations=5))
    f_outputs = write_v22f_meta_router_synthesis(
        V22FConfig(
            report_root=f_root,
            v22a_root=a_root,
            v22b_root=b_root,
            v22c_root=c_root,
            v22d_root=d_root,
            v22e_root=e_root,
        )
    )

    assert pd.read_csv(c_outputs["policy_vs_benchmark"]).shape[0] >= 3
    assert pd.read_csv(d_outputs["threshold_surface"]).shape[0] >= 9
    assert not pd.read_csv(e_outputs["negative_control_summary"]).empty
    checklist = pd.read_csv(f_outputs["promotion_checklist"])
    assert "random_p75_control_passed" in checklist["requirement"].tolist()
