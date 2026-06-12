from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v10c_burst_phase_allocation import (
    _add_asof_burst_phase,
    _burst_tranche_allocation_summary,
    _simulate_capacity_rule,
)


def _sample_trades() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for idx in range(12):
        entry = base + pd.Timedelta(minutes=5 * idx)
        late = idx >= 8
        rows.append(
            {
                "row_id": idx,
                "exchange": "bybit",
                "symbol": f"S{idx}USDT",
                "candidate": "CIC1_beta_extreme" if late else "CIC2_beta_broad",
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "cost_single_side_bps": 20.0,
                "net_return": 0.04 if late else -0.005,
                "month": "2026-01",
                "holding_minutes": 240.0,
                "rank_beta_extreme_strength": float(idx),
                "rank_local_volume_shock_strength": float(idx),
            }
        )
    return pd.DataFrame(rows)


def test_asof_burst_phase_uses_count_so_far_not_final_size() -> None:
    phased = _add_asof_burst_phase(_sample_trades(), "1h")

    assert phased["burst_count_so_far"].head(4).tolist() == [1, 2, 3, 4]
    assert phased["final_burst_size"].iloc[0] == 12
    assert not phased["uses_final_burst_size_for_decision"].any()
    assert phased.loc[8, "burst_phase_bucket"] == "order_9_14"


def test_dynamic_capacity_rule_applies_late_phase_cap() -> None:
    phased = _add_asof_burst_phase(_sample_trades(), "1h")

    selected, skipped = _simulate_capacity_rule(
        phased,
        rule="Ramp_A_3_5_8",
        cap_fn=lambda row: 3 if row["burst_count_so_far"] <= 3 else (5 if row["burst_count_so_far"] <= 8 else 8),
    )

    assert len(selected) == 8
    assert selected["burst_count_so_far"].ge(9).sum() == 3
    assert not skipped.empty


def test_late_heavy_tranche_beats_early_heavy_when_late_returns_are_better() -> None:
    phased = _add_asof_burst_phase(_sample_trades(), "1h")

    summary = _burst_tranche_allocation_summary(phased)
    early = summary[summary["scheme"].eq("Tranche_30_30_40")].iloc[0]
    late = summary[summary["scheme"].eq("Tranche_10_30_60")].iloc[0]

    assert late["portfolio_burst_net20"] > early["portfolio_burst_net20"]
