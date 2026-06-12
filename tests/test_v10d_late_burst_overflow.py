from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v10d_late_burst_overflow import (
    POLICIES,
    _overflow_reports,
    _overflow_stress_summary,
    _simulate_overflow_policy,
)


def _sample_pool() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for idx in range(10):
        late = idx >= 8
        entry = base + pd.Timedelta(minutes=5 * idx)
        rows.append(
            {
                "row_id": idx,
                "exchange": "bybit",
                "symbol": f"S{idx}USDT",
                "candidate": "CIC1_beta_extreme" if late else "CIC2_beta_broad",
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "cost_single_side_bps": 20.0,
                "net_return": 0.04 if late else 0.01,
                "month": "2026-01",
                "holding_minutes": 240.0,
            }
        )
    return _add_asof_burst_phase(pd.DataFrame(rows), "1h")


def test_overflow_policy_selects_late_candidates_after_baseline_full() -> None:
    pool = _sample_pool()
    policy = next(item for item in POLICIES if item.policy_id == "O2_late9_slots2_size050")

    ledger, skipped = _simulate_overflow_policy(pool, policy)

    assert ledger["sleeve"].eq("baseline").sum() == 8
    assert ledger["sleeve"].eq("overflow").sum() == 2
    assert ledger.loc[ledger["sleeve"].eq("overflow"), "burst_count_so_far"].min() >= 9
    assert skipped.empty


def test_additive_and_capital_neutral_are_reported_separately() -> None:
    pool = _sample_pool()

    summary, _, _, neutral = _overflow_reports(pool)
    base = summary[summary["policy_id"].eq("B0_baseline_max8_no_overflow")].iloc[0]
    o2 = summary[summary["policy_id"].eq("O2_late9_slots2_size050")].iloc[0]
    o2_neutral = neutral[neutral["policy_id"].eq("O2_late9_slots2_size050")].iloc[0]

    assert o2["portfolio_net20"] > base["portfolio_net20"]
    assert o2["capital_neutral_scale"] < 1.0
    assert o2_neutral["capital_neutral_net20"] < o2["portfolio_net20"]


def test_overflow_stress_summary_crosses_cost_size_and_slot_grid() -> None:
    pool = _sample_pool()

    stress = _overflow_stress_summary(pool)

    assert set(stress["stress_cost_single_side_bps"]) == {20, 30, 50}
    assert set(stress["overflow_max_slots"]) == {2, 4, 6}
    assert len(stress) == 27
    assert stress["incremental_return_per_extra_exposure"].notna().all()
