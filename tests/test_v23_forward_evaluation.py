from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v23_forward_evaluation import V23Config, write_v23_forward_evaluation


def test_v23_forward_evaluation_writes_decision_ledgers(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    entry = pd.Timestamp("2026-06-01T00:00:00Z")
    checkpoint = pd.DataFrame(
        [
            {
                "portfolio_id": "P2_MAX8_BASELINE",
                "trade_id": "t1",
                "signal_id": "s1",
                "symbol": "AAAUSDT",
                "entry_time": entry,
                "selected": True,
                "is_core": True,
                "is_overflow": False,
                "position_size": 1.0,
                "concurrent_positions": 0,
                "effective_net_return_10bp": 0.012,
                "effective_net_return_20bp": 0.010,
                "effective_net_return_30bp": 0.008,
                "net_return_20bp": 0.010,
                "volume_impulse_density_at_entry": 0.6,
                "cluster_impulse_density_at_entry": 0.5,
                "burst_count_so_far": 10,
            },
            {
                "portfolio_id": "P2_MAX8_CP60",
                "trade_id": "t2",
                "signal_id": "s2",
                "symbol": "BBBUSDT",
                "entry_time": entry + pd.Timedelta(minutes=15),
                "checkpoint_time": entry + pd.Timedelta(minutes=75),
                "selected": True,
                "is_core": True,
                "is_overflow": False,
                "position_size": 1.0,
                "concurrent_positions": 1,
                "checkpoint_triggered": True,
                "checkpoint_net20": -0.01,
                "net_if_checkpoint_exit_20bp": -0.012,
                "net_if_kept_counterfactual_20bp": -0.05,
                "effective_net_return_10bp": -0.010,
                "effective_net_return_20bp": -0.012,
                "effective_net_return_30bp": -0.014,
                "volume_impulse_density_at_entry": 0.2,
                "cluster_impulse_density_at_entry": 0.1,
                "burst_count_so_far": 1,
                "beta_high_protection": False,
            },
            {
                "portfolio_id": "P2_MAX8_CP60_PROTECT_A_CAP2",
                "trade_id": "t3",
                "signal_id": "s3",
                "symbol": "CCCUSDT",
                "entry_time": entry + pd.Timedelta(minutes=30),
                "selected": True,
                "is_core": True,
                "is_overflow": False,
                "position_size": 1.0,
                "concurrent_positions": 2,
                "cp60_would_exit": True,
                "beta_high_protection": True,
                "protected_by_beta_high": True,
                "counterfactual_cp60_exit_net20": -0.02,
                "actual_keep_exit_net20": 0.03,
                "delta_vs_cp60": 0.05,
                "slot_blocked_minutes": 20,
                "missed_trade_due_to_protection": False,
                "effective_net_return_10bp": 0.032,
                "effective_net_return_20bp": 0.030,
                "effective_net_return_30bp": 0.028,
                "net_if_kept_counterfactual_20bp": 0.03,
            },
        ]
    )
    checkpoint.to_parquet(source / "checkpoint_trade_ledger.parquet", index=False)
    pd.DataFrame(
        [
            {
                "trade_id": "o1",
                "burst_count_so_far": 9,
                "is_overflow": True,
                "position_size": 0.5,
                "net_return_20bp": 0.04,
                "extra_exposure": 0.5,
            }
        ]
    ).to_parquet(source / "overflow_trade_ledger.parquet", index=False)
    pd.DataFrame(
        [
            {
                "signal_id": "s1",
                "logistic_no_trade_prob": 0.2,
            }
        ]
    ).to_parquet(source / "pre_entry_router_counterfactual_live.parquet", index=False)

    outputs = write_v23_forward_evaluation(V23Config(report_root=tmp_path / "report", source_root=source))

    assert outputs["live_architecture_summary"].exists()
    arch = pd.read_csv(outputs["live_architecture_summary"])
    assert {"P2_MAX8_BASELINE", "P2_MAX8_CP60", "P2_MAX8_CP60_PROTECT_A_CAP2"}.issubset(set(arch["structure"]))
    cp60 = pd.read_csv(outputs["cp60_live_attribution"])
    assert bool(cp60.iloc[0]["true_good_exit"])
    overflow = pd.read_csv(outputs["overflow_live_attribution"])
    assert float(overflow.iloc[0]["incremental_vs_core"]) > 0
    protect = pd.read_csv(outputs["protect_a_counterfactual_live"])
    assert bool(protect.iloc[0]["would_protect"])
    regime = pd.read_csv(outputs["live_regime_diagnostics"])
    assert "low_coimpulse_score" in regime.columns
