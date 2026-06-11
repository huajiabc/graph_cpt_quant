from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v09d1 import (
    _basket_max8_baseline,
    _burst_delayed_allocation,
    _portfolio_rows_for_variants,
)


def _trade_rows() -> pd.DataFrame:
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
                "exit_time": entry + pd.Timedelta(hours=4),
                "exit_variant": "E0_vol_regime_fast",
                "cost_single_side_bps": 20.0,
                "net_return": 0.01 + idx * 0.001,
                "month": "2026-01",
                "holding_minutes": 240.0,
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


def test_v09d1_builds_max8_true_exit_baseline() -> None:
    trades = _trade_rows()

    summary, timeline, skipped = _portfolio_rows_for_variants(
        trades,
        ["E0_vol_regime_fast"],
        architecture="true_time_stop",
        rule_prefix="basket_first_come",
    )
    baseline = _basket_max8_baseline(summary)

    assert not baseline.empty
    row = baseline.iloc[0]
    assert row["max_positions"] == "8"
    assert row["selected_trades"] == 8
    assert row["skipped_trades"] == 4
    assert not timeline.empty
    assert not skipped.empty


def test_v09d1_burst_delayed_allocation_is_diagnostic() -> None:
    trades = _trade_rows()

    delayed = _burst_delayed_allocation(trades)

    assert not delayed.empty
    assert delayed["architecture"].eq("burst_delayed_allocation_diagnostic").all()
    assert delayed["notes"].str.contains("delayed entry price is not repriced", case=False).all()
