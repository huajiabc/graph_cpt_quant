from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v13d_cp60_context_protection import (
    _add_context_features,
    _add_high_flags,
    _thresholds_from_exits,
)
from pressure_graph.reports.v13e_cp60_beta_protection_stability import (
    _leave_one_protected_exit_out,
    _protect_a_ledger,
    _protected_distribution,
    _protected_exit_ledger,
    _protection_o6_integration,
)


def _row(idx: int, *, net_keep: float, checkpoint_net: float, beta: float) -> dict[str, object]:
    entry = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=5 * idx)
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
        "cluster_impulse_density": 0.6 if beta >= 99 else 0.1,
        "volume_z_1h": 3.0,
        "burst_id": "b0",
        "burst_count_so_far": idx + 1,
        "month": "2026-01",
    }


def _sample() -> pd.DataFrame:
    data = pd.DataFrame(
        [
            _row(0, net_keep=0.04, checkpoint_net=-0.005, beta=100),
            _row(1, net_keep=-0.05, checkpoint_net=-0.010, beta=99),
            _row(2, net_keep=-0.02, checkpoint_net=-0.003, beta=95),
            _row(3, net_keep=0.03, checkpoint_net=-0.004, beta=90),
        ]
    )
    data = _add_context_features(data)
    return _add_high_flags(data, _thresholds_from_exits(data))


def test_beta_protection_stability_tables() -> None:
    sample = _sample()

    cp_ledger, cp_skipped, protect_ledger, protect_skipped = _protect_a_ledger(sample)
    loo = _leave_one_protected_exit_out(sample)
    protected = _protected_exit_ledger(sample)
    distribution = _protected_distribution(protected)
    o6 = _protection_o6_integration(sample)

    assert not cp_ledger.empty
    assert cp_skipped.empty
    assert not protect_ledger.empty
    assert protect_skipped.empty
    assert not loo.empty
    assert "still_above_CP60" in loo.columns
    assert not protected.empty
    assert {"month", "symbol", "burst_id"}.issubset(set(distribution["group_col"]))
    assert {"S3_CP60_O6", "S3_Protect_A_O6"}.issubset(set(o6["structure"]))
