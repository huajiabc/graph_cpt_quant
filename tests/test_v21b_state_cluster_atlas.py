from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v21b_state_cluster_atlas import (
    V21BConfig,
    _action_summary,
    _prepare_membership,
    _state_cluster_summary,
)


def _row(idx: int, *, candidate: str, net: float, checkpoint_net: float, burst_count: int) -> dict[str, object]:
    entry = pd.Timestamp("2025-10-01T00:00:00Z") + pd.Timedelta(hours=idx)
    return {
        "symbol": f"S{idx}USDT",
        "candidate": candidate,
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=4),
        "checkpoint_time": entry + pd.Timedelta(hours=1),
        "checkpoint_price_covered": True,
        "checkpoint_net_at_cost": checkpoint_net,
        "net_return_at_cost": net,
        "month": entry.strftime("%Y-%m"),
        "btc_market_state": "BTC_up" if idx % 2 == 0 else "BTC_chop",
        "market_impulse_density": 0.1 + idx * 0.03,
        "cluster_impulse_density": 0.2 + idx * 0.02,
        "c2_beta_extension_score": 80 + idx,
        "volume_z_1h": 1.0 + idx * 0.1,
        "ret_4h": 0.01 * idx,
        "ret_4h_percentile": 50 + idx,
        "burst_count_so_far": burst_count,
        "mfe_12h": max(net, 0.0) + 0.02,
        "mae_12h": min(net, 0.0) - 0.01,
        "mfe_24h": max(net, 0.0) + 0.03,
        "mae_24h": min(net, 0.0) - 0.02,
        "hit_10pct_12h": net > 0.05,
        "hit_20pct_24h": False,
        "trade_key": f"sig-{idx}",
        "signal_id": f"sig-{idx}",
    }


def test_state_cluster_atlas_membership_and_action_summary() -> None:
    sample = pd.DataFrame(
        [
            _row(0, candidate="CIC1_beta_extreme", net=0.05, checkpoint_net=-0.01, burst_count=1),
            _row(1, candidate="CIC2_beta_broad", net=-0.02, checkpoint_net=-0.03, burst_count=2),
            _row(2, candidate="CIC1_beta_extreme", net=0.08, checkpoint_net=0.01, burst_count=9),
            _row(3, candidate="CIC2_beta_broad", net=0.01, checkpoint_net=-0.005, burst_count=10),
            _row(4, candidate="CIC1_beta_extreme", net=-0.03, checkpoint_net=-0.02, burst_count=3),
            _row(5, candidate="CIC2_beta_broad", net=0.04, checkpoint_net=0.002, burst_count=11),
        ]
    )

    membership = _prepare_membership(sample, V21BConfig(n_clusters=3))
    assert "state_cluster_id" in membership.columns
    assert membership["state_cluster_id"].nunique() <= 3
    assert membership["late_burst_o6_candidate"].sum() == 3

    summary = _state_cluster_summary(membership)
    assert not summary.empty
    assert {"trades", "net20_sum", "hit10_12h", "cp60_would_exit_rate"}.issubset(summary.columns)

    action = _action_summary(membership)
    assert set(action["action"]) == {
        "CP60_if_no_followthrough",
        "Protect_A_beta_high",
        "O6_late_burst_overflow",
    }
