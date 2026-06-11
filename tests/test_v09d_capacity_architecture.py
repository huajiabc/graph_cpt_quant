from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v09d import (
    _basket_capacity_curve,
    _capital_lock_summary,
    _replacement_rule_summary,
    _reserve_capacity_summary,
)


def _sample_trades() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for idx in range(12):
        entry = base + pd.Timedelta(minutes=15 * idx)
        rows.append(
            {
                "exchange": "bybit",
                "symbol": f"S{idx}USDT",
                "candidate": "CIC1_beta_extreme" if idx < 7 else "CIC2_beta_broad",
                "signal_time": entry - pd.Timedelta(minutes=15),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=3),
                "cost_single_side_bps": 20.0,
                "net_return": 0.01 + idx * 0.001,
                "month": "2026-01",
                "holding_minutes": 180.0,
                "c2_beta_extension_bucket": "beta_extreme_overextended" if idx < 7 else "beta_extended",
                "c2_beta_extension_score": float(idx),
                "volume_z_1h": float(idx + 1),
                "volume_impulse_density": 0.4,
                "cluster_impulse_density": float(idx) / 10.0,
                "dynamic_all_rank": idx + 1,
                "bars_from_signal_to_entry": 2,
                "base_signal_id": f"bybit|S{idx}USDT|{entry.isoformat()}",
                "rank_beta_extreme_strength": float(idx),
                "rank_local_volume_shock_strength": float(idx + 1),
                "rank_market_impulse_density": 0.4,
                "rank_cluster_impulse_density": float(idx) / 10.0,
                "rank_liquidity": float(-idx),
                "rank_reclaim_quality": -2.0,
                "rank_composite_simple": float(idx) / 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_basket_capacity_curve_uses_finite_capacity_units() -> None:
    trades = _sample_trades()

    basket, timeline, skipped = _basket_capacity_curve(trades)

    row = basket[
        basket["pool"].eq("P0_CIC1_ONLY")
        & basket["max_positions"].astype(str).eq("5")
    ].iloc[0]
    assert row["selected_trades"] == 5
    assert row["capital_units"] == 5
    assert row["portfolio_net20"] > 0
    assert not timeline.empty
    assert not skipped.empty


def test_reserve_capacity_can_skip_early_burst_trades() -> None:
    trades = _sample_trades()

    reserve = _reserve_capacity_summary(trades)

    row = reserve[
        reserve["pool"].eq("P2_CIC1_CIC2_COMBINED")
        & reserve["max_positions"].astype(str).eq("10")
        & reserve["rule"].eq("reserve_50pct_phase_60m")
    ].iloc[0]
    assert row["selected_trades"] < len(trades)
    assert row["skipped_trades"] > 0


def test_replacement_and_capital_lock_reports_are_marked_diagnostic() -> None:
    trades = _sample_trades()

    replacement = _replacement_rule_summary(trades)
    lock = _capital_lock_summary(trades)

    assert replacement["architecture"].eq("replacement_rule").all()
    assert replacement["notes"].str.contains("diagnostic", case=False).all()
    assert lock["architecture"].eq("capital_lock_diagnostic").all()
    assert lock["notes"].str.contains("capacity release diagnostics", case=False).all()
