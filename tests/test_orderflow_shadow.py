from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.orderflow import (
    aggregate_orderflow_1m,
    attach_orderflow_to_trades,
    build_orderflow_demand_queue,
    select_orderflow_symbols,
    summarize_trade_window,
)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "timestamp": "2026-06-06 00:00:05Z",
                "execId": "a",
                "price": 100.0,
                "size": 2.0,
                "turnover": 200.0,
                "side": "Buy",
            },
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "timestamp": "2026-06-06 00:00:30Z",
                "execId": "b",
                "price": 101.0,
                "size": 1.0,
                "turnover": 101.0,
                "side": "Sell",
            },
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "timestamp": "2026-06-06 00:01:02Z",
                "execId": "c",
                "price": 102.0,
                "size": 3.0,
                "turnover": 306.0,
                "side": "Buy",
            },
        ]
    )


def test_aggregate_orderflow_1m_splits_buy_sell_turnover() -> None:
    agg = aggregate_orderflow_1m(_trades(), large_trade_quantile=0.5)

    first = agg[agg["bar_open_time"].eq(pd.Timestamp("2026-06-06 00:00:00Z"))].iloc[0]
    assert first["trade_count"] == 2
    assert first["buy_trade_count"] == 1
    assert first["sell_trade_count"] == 1
    assert first["buy_turnover"] == pytest.approx(200.0)
    assert first["sell_turnover"] == pytest.approx(101.0)
    assert first["taker_buy_ratio"] == pytest.approx(200.0 / 301.0)
    assert first["cvd_delta_turnover"] == pytest.approx(99.0)


def test_summarize_trade_window_uses_exact_time_bounds() -> None:
    stats = summarize_trade_window(
        _trades(),
        pd.Timestamp("2026-06-06 00:00:00Z"),
        pd.Timestamp("2026-06-06 00:01:00Z"),
        large_trade_quantile=0.5,
    )

    assert stats["covered"]
    assert stats["trade_count"] == 2
    assert stats["turnover"] == pytest.approx(301.0)
    assert stats["taker_buy_ratio"] == pytest.approx(200.0 / 301.0)


def test_attach_orderflow_to_trades_marks_missing_and_covered_windows() -> None:
    paper_trades = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "signal_id": "s1",
                "candidate": "CIC1_FILTERED_MIR1",
                "candidate_role": "primary",
                "baseline_kind": "",
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "local_volume_shock_time": pd.Timestamp("2026-06-06 00:00:00Z"),
                "pullback_time": pd.Timestamp("2026-06-06 00:00:00Z"),
                "reclaim_time": pd.Timestamp("2026-06-06 00:01:00Z"),
                "entry_time": pd.Timestamp("2026-06-06 00:01:00Z"),
                "portfolio_accepted": True,
            }
        ]
    )

    shadow, windows = attach_orderflow_to_trades(paper_trades, {"AAAUSDT": _trades()})

    assert len(shadow) == 1
    assert len(windows) == 6
    row = shadow.iloc[0]
    assert row["shock_bar_covered"]
    assert row["reclaim_bar_covered"]
    assert row["shock_bar_trade_count"] == 3
    assert row["reclaim_bar_trade_count"] == 1


def test_build_orderflow_demand_queue_prioritizes_event_windows(tmp_path) -> None:
    source = tmp_path / "reports"
    source.mkdir()
    pd.DataFrame(
        [
            {
                "trade_id": "p0",
                "signal_id": "s0",
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "candidate": "CIC1_FILTERED_MIR1",
                "candidate_role": "primary",
                "local_volume_shock_time": pd.Timestamp("2026-06-06 00:00:00Z"),
                "pullback_time": pd.Timestamp("2026-06-06 00:15:00Z"),
                "reclaim_time": pd.Timestamp("2026-06-06 00:30:00Z"),
                "entry_time": pd.Timestamp("2026-06-06 00:30:00Z"),
                "exit_time": pd.Timestamp("2026-06-06 01:30:00Z"),
                "portfolio_accepted": True,
            },
            {
                "trade_id": "p1",
                "signal_id": "s1",
                "exchange": "bybit",
                "symbol": "BBBUSDT",
                "candidate": "CIC2_FILTERED_MIR1",
                "candidate_role": "secondary_shadow",
                "local_volume_shock_time": pd.Timestamp("2026-06-06 00:00:00Z"),
                "entry_time": pd.Timestamp("2026-06-06 00:45:00Z"),
                "exit_time": pd.Timestamp("2026-06-06 01:30:00Z"),
                "portfolio_accepted": False,
                "portfolio_skip_reason": "portfolio_full",
            },
        ]
    ).to_parquet(source / "paper_trades.parquet", index=False)

    queue = build_orderflow_demand_queue(
        source,
        tmp_path / "orderflow",
        tmp_path / "demand.parquet",
        report_lookback_days=30,
        core_reference_symbols=(),
        now=pd.Timestamp("2026-06-07 00:00:00Z"),
    )

    assert list(queue["priority"]) == ["P0", "P1"]
    assert queue.loc[0, "reason"] == "cic_primary_selected_trade"
    assert queue.loc[1, "reason"] == "capacity_skipped_candidate"
    assert queue["window_start"].notna().all()
    assert queue["window_end"].notna().all()


def test_select_orderflow_symbols_prefers_demand_queue_over_feature_topn(tmp_path) -> None:
    features = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "feature_time": pd.Timestamp("2026-06-06 00:00:00Z"),
                "dynamic_all_rank": 1,
            },
            {
                "symbol": "ETHUSDT",
                "feature_time": pd.Timestamp("2026-06-06 00:00:00Z"),
                "dynamic_all_rank": 2,
            },
        ]
    )
    feature_path = tmp_path / "features.parquet"
    features.to_parquet(feature_path, index=False)
    queue = pd.DataFrame(
        [
            {
                "symbol": "AAVEUSDT",
                "priority": "P0",
                "status": "pending",
                "window_end": pd.Timestamp("2026-06-06 01:00:00Z"),
            }
        ]
    )

    selected = select_orderflow_symbols(
        feature_path,
        tmp_path / "missing_reports",
        top_n=50,
        max_symbols=2,
        demand_queue=queue,
    )

    assert selected == ["AAVEUSDT", "BTCUSDT"]
