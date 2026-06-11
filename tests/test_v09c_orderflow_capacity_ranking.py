from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v09c import write_v09c_orderflow_capacity_ranking


def _trade(
    idx: int,
    candidate: str,
    entry: str,
    net20: float,
    reclaim_ratio: float | None,
    reclaim_cvd: float | None,
) -> dict[str, object]:
    return {
        "trade_id": f"t{idx}",
        "signal_id": f"s{idx}",
        "candidate": candidate,
        "candidate_role": "primary" if candidate == "CIC1_FILTERED_MIR1" else "secondary_shadow",
        "symbol": f"AAA{idx % 3}USDT",
        "entry_time": entry,
        "exit_time": pd.Timestamp(entry) + pd.Timedelta(hours=4),
        "market_gate_time": pd.Timestamp(entry) - pd.Timedelta(minutes=45),
        "holding_minutes": 240.0,
        "net_return_20bp": net20,
        "portfolio_accepted": False,
        "shock_bar_covered": reclaim_ratio is not None,
        "pullback_window_covered": reclaim_ratio is not None,
        "reclaim_bar_covered": reclaim_ratio is not None,
        "entry_bar_covered": reclaim_ratio is not None,
        "post_entry_15m_covered": reclaim_ratio is not None,
        "post_entry_1h_covered": reclaim_ratio is not None,
        "shock_bar_taker_buy_ratio": 0.50 if reclaim_ratio is not None else None,
        "shock_bar_cvd_delta_turnover": 1.0 if reclaim_ratio is not None else None,
        "shock_bar_large_buy_turnover": 10.0 if reclaim_ratio is not None else None,
        "shock_bar_large_sell_turnover": 2.0 if reclaim_ratio is not None else None,
        "pullback_window_cvd_delta_turnover": 1.0 if reclaim_ratio is not None else None,
        "pullback_window_buy_sell_imbalance": 0.1 if reclaim_ratio is not None else None,
        "reclaim_bar_taker_buy_ratio": reclaim_ratio,
        "reclaim_bar_cvd_delta_turnover": reclaim_cvd,
        "reclaim_bar_buy_sell_imbalance": reclaim_cvd,
        "reclaim_bar_large_buy_count": 2 if reclaim_ratio is not None else 0,
        "reclaim_bar_large_sell_count": 0,
        "reclaim_bar_large_buy_turnover": 20.0 if reclaim_ratio is not None else 0.0,
        "reclaim_bar_large_sell_turnover": 1.0,
        "post_entry_15m_cvd_delta_turnover": reclaim_cvd,
        "post_entry_1h_cvd_delta_turnover": reclaim_cvd,
    }


def test_v09c_marks_insufficient_orderflow_coverage(tmp_path) -> None:
    source = tmp_path / "source"
    report = tmp_path / "report"
    source.mkdir()
    pd.DataFrame(
        [
            _trade(1, "CIC1_FILTERED_MIR1", "2026-06-01 00:00:00Z", 0.01, None, None),
            _trade(2, "CIC2_FILTERED_MIR1", "2026-06-01 00:15:00Z", -0.01, None, None),
        ]
    ).to_csv(source / "orderflow_shadow_trades.csv", index=False)

    outputs = write_v09c_orderflow_capacity_ranking(source, report)

    notes = outputs["candidate_notes"].read_text(encoding="utf-8")
    assert "sample_status: insufficient_orderflow_coverage" in notes
    coverage = pd.read_csv(outputs["orderflow_feature_coverage"])
    assert coverage["reclaim_bar_covered_trades"].sum() == 0


def test_v09c_writes_orderflow_ranking_summary(tmp_path) -> None:
    source = tmp_path / "source"
    report = tmp_path / "report"
    source.mkdir()
    rows = []
    for idx in range(12):
        rows.append(
            _trade(
                idx,
                "CIC1_FILTERED_MIR1" if idx % 2 == 0 else "CIC2_FILTERED_MIR1",
                f"2026-06-01 {idx:02d}:00:00Z",
                0.02 if idx % 3 == 0 else -0.005,
                0.70 if idx % 3 == 0 else 0.40,
                5.0 if idx % 3 == 0 else -1.0,
            )
        )
    pd.DataFrame(rows).to_csv(source / "orderflow_shadow_trades.csv", index=False)

    outputs = write_v09c_orderflow_capacity_ranking(source, report)

    ranking = pd.read_csv(outputs["orderflow_ranking_summary"])
    assert {"R2_reclaim_taker_buy_ratio", "R8_simple_orderflow_composite"}.issubset(
        set(ranking["ranking_rule"])
    )
    assert (ranking["selected_trades"] > 0).any()
