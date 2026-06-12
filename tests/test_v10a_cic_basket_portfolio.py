from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v10a_cic_basket_portfolio import _burst_basket_summary, _capacity_curve


def _sample_trades() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for idx in range(12):
        entry = base + pd.Timedelta(minutes=15 * idx)
        is_cic1 = idx < 6
        rows.append(
            {
                "exchange": "bybit",
                "symbol": f"S{idx}USDT",
                "candidate": "CIC1_beta_extreme" if is_cic1 else "CIC2_beta_broad",
                "signal_time": entry - pd.Timedelta(minutes=15),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=3),
                "cost_single_side_bps": 20.0,
                "net_return": 0.03 if is_cic1 else 0.005,
                "month": "2026-01",
                "holding_minutes": 180.0,
                "base_signal_id": f"bybit|S{idx}USDT|{entry.isoformat()}",
                "rank_first_come_first_served": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_v10a_capacity_curve_tracks_p2_max8_selected_and_skipped() -> None:
    trades = _sample_trades()

    capacity, timeline, skipped = _capacity_curve(trades)

    row = capacity[
        capacity["pool"].eq("P2_CIC1_CIC2_COMBINED")
        & capacity["max_positions"].astype(str).eq("8")
    ].iloc[0]
    assert row["selected_trades"] == 8
    assert row["skipped_trades"] == 4
    assert row["portfolio_net20"] > 0
    assert not timeline.empty
    assert not skipped.empty


def test_v10a_burst_budget_reports_cic_weight_effect() -> None:
    trades = _sample_trades()

    summary, detail, trade_detail = _burst_basket_summary(trades)

    p2 = summary[
        summary["pool"].eq("P2_CIC1_CIC2_COMBINED")
        & summary["burst_window"].eq("1h")
        & summary["max_per_burst"].astype(str).eq("unlimited")
    ]
    equal = p2[p2["rule"].eq("cic_1_to_1_fixed_burst_budget")].iloc[0]
    cic3 = p2[p2["rule"].eq("cic_3_to_1_fixed_burst_budget")].iloc[0]
    assert cic3["avg_burst_return_net20"] > equal["avg_burst_return_net20"]
    assert not detail.empty
    assert not trade_detail.empty
