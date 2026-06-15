from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v13d_cp60_context_protection import (
    _add_high_flags,
    _add_context_features,
    _context_cross_table,
    _protection_counterfactual,
    _thresholds_from_exits,
)


def _row(idx: int, *, net_keep: float, checkpoint_net: float, beta: float, cluster: float) -> dict[str, object]:
    entry = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=idx * 5)
    return {
        "exchange": "bybit",
        "symbol": f"S{idx}USDT",
        "candidate": "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad",
        "signal_id": f"sig-{idx}",
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=4),
        "checkpoint_time": entry + pd.Timedelta(hours=1),
        "checkpoint_price_covered": True,
        "entry_price": 100.0,
        "checkpoint_gross_return": checkpoint_net + 0.004,
        "checkpoint_net_at_cost": checkpoint_net,
        "checkpoint_mfe": 0.01,
        "checkpoint_mae": -0.01,
        "gross_return": net_keep + 0.004,
        "net_return_at_cost": net_keep,
        "volume_impulse_density": 0.4,
        "c2_beta_extension_score": beta,
        "cluster_impulse_density": cluster,
        "volume_z_1h": 3.0,
        "burst_id": "b0",
    }


def test_context_protection_counterfactual_outputs() -> None:
    sample = pd.DataFrame(
        [
            _row(0, net_keep=0.04, checkpoint_net=-0.005, beta=100, cluster=0.8),
            _row(1, net_keep=-0.05, checkpoint_net=-0.010, beta=99, cluster=0.7),
            _row(2, net_keep=-0.02, checkpoint_net=-0.003, beta=95, cluster=0.1),
            _row(3, net_keep=0.03, checkpoint_net=-0.004, beta=90, cluster=0.2),
        ]
    )
    sample = _add_context_features(sample)
    sample = _add_high_flags(sample, _thresholds_from_exits(sample))

    context = _context_cross_table(sample, neutral_delta=0.001)
    protection, ledger, skipped = _protection_counterfactual(sample, neutral_delta=0.001)

    assert not context.empty
    assert not protection.empty
    assert not ledger.empty
    assert skipped.empty
    beta_row = protection[protection["rule"].eq("Protect_A_beta_high")].iloc[0]
    assert beta_row["protected_cp60_exits"] >= 1
    assert "false_exits_saved" in protection.columns
    assert "true_good_exits_lost" in protection.columns
