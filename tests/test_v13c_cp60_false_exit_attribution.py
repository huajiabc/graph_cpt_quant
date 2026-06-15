from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v13c_cp60_false_exit_attribution import (
    _categorical_summary,
    _classify_cp60_exits,
    _feature_bucket_summary,
    _summary_by_class,
)


def test_cp60_false_exit_classification_and_summaries() -> None:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    ledger = pd.DataFrame(
        [
            {
                "signal_id": "a",
                "symbol": "AAAUSDT",
                "candidate": "CIC1_beta_extreme",
                "entry_time": base,
                "checkpoint_time": base + pd.Timedelta(hours=1),
                "exit_time": base + pd.Timedelta(hours=4),
                "checkpoint_early_exit": True,
                "checkpoint_net_at_cost": -0.01,
                "checkpoint_gross_return": -0.006,
                "checkpoint_mfe": 0.01,
                "checkpoint_mae": -0.02,
                "net_return_at_cost": -0.05,
                "effective_net_return": -0.01,
                "volume_impulse_density": 0.2,
                "c2_beta_extension_score": 0.95,
                "volume_z_1h": 3.0,
                "cluster_impulse_density": 0.5,
                "btc_state_at_entry": "BTC_up",
            },
            {
                "signal_id": "b",
                "symbol": "BBBUSDT",
                "candidate": "CIC2_beta_broad",
                "entry_time": base,
                "checkpoint_time": base + pd.Timedelta(hours=1),
                "exit_time": base + pd.Timedelta(hours=4),
                "checkpoint_early_exit": True,
                "checkpoint_net_at_cost": -0.005,
                "checkpoint_gross_return": -0.001,
                "checkpoint_mfe": 0.05,
                "checkpoint_mae": -0.01,
                "net_return_at_cost": 0.04,
                "effective_net_return": -0.005,
                "volume_impulse_density": 0.3,
                "c2_beta_extension_score": 0.8,
                "volume_z_1h": 2.0,
                "cluster_impulse_density": 0.7,
                "btc_state_at_entry": "BTC_chop",
            },
            {
                "signal_id": "c",
                "symbol": "CCCUSDT",
                "candidate": "CIC2_beta_broad",
                "entry_time": base,
                "checkpoint_time": base + pd.Timedelta(hours=1),
                "exit_time": base + pd.Timedelta(hours=4),
                "checkpoint_early_exit": True,
                "checkpoint_net_at_cost": -0.002,
                "checkpoint_gross_return": 0.002,
                "checkpoint_mfe": 0.02,
                "checkpoint_mae": -0.01,
                "net_return_at_cost": -0.0025,
                "effective_net_return": -0.002,
                "volume_impulse_density": 0.4,
                "c2_beta_extension_score": 0.7,
                "volume_z_1h": 1.5,
                "cluster_impulse_density": 0.4,
                "btc_state_at_entry": "BTC_up",
            },
        ]
    )

    exits = _classify_cp60_exits(ledger, neutral_delta=0.001)
    assert exits["exit_class"].tolist() == ["true_good_exit", "false_exit", "neutral_exit"]
    assert exits["cic_type"].tolist() == ["CIC1", "CIC2", "CIC2"]

    summary = _summary_by_class(exits)
    assert set(summary["exit_class"]) == {"true_good_exit", "false_exit", "neutral_exit"}

    buckets = _feature_bucket_summary(exits)
    assert "checkpoint_net" in set(buckets["feature"])

    categorical = _categorical_summary(exits)
    assert {"cic_type", "btc_state"}.issubset(set(categorical["feature"]))
