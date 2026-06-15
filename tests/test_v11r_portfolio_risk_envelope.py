from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v11r_portfolio_risk_envelope import RiskPolicy, _risk_reports, simulate_risk_policy


def _sample_pool() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for idx in range(10):
        entry = base + pd.Timedelta(minutes=5 * idx)
        rows.append(
            {
                "row_id": idx,
                "exchange": "bybit",
                "symbol": f"S{idx}USDT",
                "candidate": "CIC1_beta_extreme",
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "cost_single_side_bps": 20.0,
                "net_return": 0.04 if idx >= 8 else 0.01,
                "month": "2026-01",
                "holding_minutes": 240.0,
            }
        )
    return _add_asof_burst_phase(pd.DataFrame(rows), "1h")


def test_o6_risk_policy_adds_late_overflow_after_core_is_full() -> None:
    pool = _sample_pool()
    policy = RiskPolicy("o6", "core_plus_o6", True, overflow_max_slots=4)

    ledger, skipped = simulate_risk_policy(pool, policy)

    assert ledger["sleeve"].eq("core").sum() == 8
    assert ledger["sleeve"].eq("overflow").sum() == 2
    assert ledger.loc[ledger["sleeve"].eq("overflow"), "burst_count_so_far"].min() >= 9
    assert ledger.loc[ledger["sleeve"].eq("overflow"), "exposure_weight"].tolist() == [0.5, 0.5]
    assert skipped.empty


def test_total_exposure_cap_blocks_or_allows_overflow() -> None:
    pool = _sample_pool()
    cap8 = RiskPolicy("cap8", "total_exposure_cap", True, overflow_max_slots=4, total_exposure_cap=8.0)
    cap9 = RiskPolicy("cap9", "total_exposure_cap", True, overflow_max_slots=4, total_exposure_cap=9.0)

    ledger8, skipped8 = simulate_risk_policy(pool, cap8)
    ledger9, skipped9 = simulate_risk_policy(pool, cap9)

    assert ledger8["sleeve"].eq("overflow").sum() == 0
    assert skipped8["skip_reason"].eq("total_exposure_cap").sum() == 2
    assert ledger9["sleeve"].eq("overflow").sum() == 2
    assert skipped9.empty


def test_risk_reports_include_core_o6_and_cap_sweeps() -> None:
    pool = _sample_pool()

    summary, ledger, skipped, period, burst = _risk_reports(pool)

    assert {"core_p2_max8", "core_p2_max8_plus_o6"}.issubset(set(summary["policy_id"]))
    assert summary["policy_type"].isin(["total_exposure_cap", "daily_new_exposure_cap", "rolling_4h_new_exposure_cap"]).any()
    assert not ledger.empty
    assert not skipped.empty
    assert not period.empty
    assert not burst.empty
    o6 = summary[summary["policy_id"].eq("core_p2_max8_plus_o6")].iloc[0]
    assert o6["overflow_trades"] == 2
    assert o6["max_total_exposure"] == 9.0
