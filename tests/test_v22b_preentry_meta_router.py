from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v22b_preentry_meta_router import (
    V22BConfig,
    write_v22b_preentry_meta_router,
)


def _row(idx: int, month: str, label: str) -> dict[str, object]:
    entry = pd.Timestamp(f"{month}-02T00:00:00Z") + pd.Timedelta(minutes=15 * idx)
    exit_time = entry + pd.Timedelta(hours=4)
    net = 0.02 if label == "core_trade" else -0.02
    return {
        "trade_key": f"t{idx}",
        "symbol": f"S{idx % 7}USDT",
        "candidate": "CIC1_beta_extreme" if idx % 2 else "CIC2_beta_broad",
        "entry_time": entry.isoformat(),
        "entry_month": month,
        "period": "search" if month < "2026-02" else "validation",
        "exit_time": exit_time.isoformat(),
        "checkpoint_time": (entry + pd.Timedelta(hours=1)).isoformat(),
        "checkpoint_price_covered": True,
        "checkpoint_net_at_cost": -0.005 if label == "no_trade" else 0.005,
        "net_return_at_cost": net,
        "net20": net,
        "pre_entry_action_label": label,
        "post_entry_checkpoint_label": "cp60_exit_better" if label == "no_trade" else "not_cp60_eligible",
        "protect_a_label": "not_protect_eligible",
        "capacity_overflow_label": "not_overflow_eligible",
        "cic_type": "CIC1" if idx % 2 else "CIC2",
        "btc_state": "BTC_up" if idx % 3 else "BTC_chop",
        "market_impulse_density": 0.7 if label == "core_trade" else 0.15,
        "cluster_density": 0.2,
        "beta_strength": 99.0 if label == "core_trade" else 88.0,
        "local_shock_strength": 4.0,
        "ret_4h": 0.05 if label == "core_trade" else 0.005,
        "ret_4h_percentile": 99.0 if label == "core_trade" else 80.0,
        "symbol_volatility_percentile": 70.0,
        "burst_count_so_far": idx % 12,
        "minutes_since_burst_start": float((idx % 4) * 15),
        "same_timestamp_peer_count": idx % 5,
        "burst_id": f"burst-{month}-{idx // 5}",
        "beta_extreme_strength_high": label == "core_trade",
    }


def test_preentry_meta_router_writes_walkforward_outputs(tmp_path: Path) -> None:
    v21g = tmp_path / "v21g"
    v21g.mkdir()
    rows = []
    idx = 0
    for month in ["2025-07", "2025-08", "2025-09", "2025-10", "2026-02", "2026-03"]:
        for inner in range(24):
            label = "core_trade" if inner % 3 else "no_trade"
            rows.append(_row(idx, month, label))
            idx += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(v21g / "meta_router_event_labels.csv", index=False)
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
    frame[feature_cols].to_csv(v21g / "meta_router_feature_matrix.csv", index=False)

    outputs = write_v22b_preentry_meta_router(
        V22BConfig(
            report_root=tmp_path / "out",
            v21g_root=v21g,
            min_train_months=2,
            min_train_events=40,
            random_permutations=5,
            logistic_steps=80,
        )
    )
    summary = pd.read_csv(outputs["walkforward_policy_summary"])
    predictions = pd.read_csv(outputs["policy_event_predictions"])
    controls = pd.read_csv(outputs["negative_controls"])

    assert "baseline_B4" in summary["policy_id"].tolist()
    assert "logistic_no_trade_t70" in summary["policy_id"].tolist()
    assert "logistic_p_no_trade" in predictions.columns
    assert len(controls) == 5
    assert outputs["candidate_notes"].exists()
