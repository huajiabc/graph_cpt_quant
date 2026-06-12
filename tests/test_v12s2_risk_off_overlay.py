from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v12s2_long_risk_off_overlay import (
    RiskOffConfig,
    _apply_market_gate,
    _apply_symbol_gate,
    _mode_metrics,
)


def _pool(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["rank_first_come_first_served"] = 0.0
    return df


def test_symbol_gate_blocks_long_after_recent_failure():
    pool = _pool(
        [
            {"symbol": "AAAUSDT", "signal_time": "2026-01-01 10:00", "entry_time": "2026-01-01 10:15",
             "exit_time": "2026-01-01 14:00", "net_return": -0.02, "holding_minutes": 225.0},
            {"symbol": "AAAUSDT", "signal_time": "2026-01-03 10:00", "entry_time": "2026-01-03 10:15",
             "exit_time": "2026-01-03 14:00", "net_return": 0.03, "holding_minutes": 225.0},
        ]
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "motif": ["S1"],
            "feature_time": pd.to_datetime(["2026-01-01 08:00"], utc=True),
        }
    )
    cfg = RiskOffConfig(symbol_cooldown_bars=32)  # 8h window
    gated = _apply_symbol_gate(pool, events, cfg)
    # First long (signal 10:00, failure at 08:00 -> 2h earlier, within 8h) is gated.
    assert bool(gated.iloc[0]) is True
    # Second long (2 days later) is outside the cooldown -> not gated.
    assert bool(gated.iloc[1]) is False


def test_symbol_gate_is_strict_as_of():
    # A failure AFTER the long signal must never gate it (no lookahead).
    pool = _pool(
        [
            {"symbol": "AAAUSDT", "signal_time": "2026-01-01 10:00", "entry_time": "2026-01-01 10:15",
             "exit_time": "2026-01-01 14:00", "net_return": 0.01, "holding_minutes": 225.0},
        ]
    )
    events = pd.DataFrame(
        {"symbol": ["AAAUSDT"], "motif": ["S1"], "feature_time": pd.to_datetime(["2026-01-01 11:00"], utc=True)}
    )
    gated = _apply_symbol_gate(pool, events, RiskOffConfig())
    assert bool(gated.iloc[0]) is False


def test_market_gate_fires_on_breadth():
    pool = _pool(
        [
            {"symbol": "ZZZUSDT", "signal_time": "2026-01-01 10:00", "entry_time": "2026-01-01 10:15",
             "exit_time": "2026-01-01 14:00", "net_return": -0.01, "holding_minutes": 225.0},
        ]
    )
    # Three distinct symbols failing within the 4h window before the signal.
    events = pd.DataFrame(
        {
            "symbol": ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
            "motif": ["S1", "S3", "S5"],
            "feature_time": pd.to_datetime(
                ["2026-01-01 09:00", "2026-01-01 09:15", "2026-01-01 09:30"], utc=True
            ),
        }
    )
    cfg = RiskOffConfig(breadth_window_bars=16, breadth_threshold=3)
    gated = _apply_market_gate(pool, events, cfg)
    assert bool(gated.iloc[0]) is True
    # Threshold of 4 should not fire with only 3 distinct symbols.
    assert bool(_apply_market_gate(pool, events, RiskOffConfig(breadth_threshold=4)).iloc[0]) is False


def test_mode_metrics_reports_gated_realized_net():
    pool = _pool(
        [
            {"symbol": f"S{i}USDT", "signal_time": f"2026-01-0{i+1} 10:00",
             "entry_time": f"2026-01-0{i+1} 10:15", "exit_time": f"2026-01-0{i+1} 14:00",
             "net_return": -0.05 if i == 0 else 0.02, "holding_minutes": 225.0, "month": "2026-01"}
            for i in range(4)
        ]
    )
    gated = pd.Series([True, False, False, False])
    metrics = _mode_metrics(pool, gated, "symbol_risk_off", max_positions=8)
    assert metrics["longs_gated"] == 1
    assert metrics["gated_realized_net_mean"] == -0.05  # removed the loser
    assert metrics["gated_loss_share"] == 1.0


def test_half_size_actually_scales_selected_gated_long():
    # A single selected gated long: half-size must halve its contribution, so the
    # half-size portfolio net must differ from the un-gated baseline. (Regression:
    # a leading-underscore marker column was silently dropped by itertuples.)
    pool = _pool(
        [
            {"symbol": "AAAUSDT", "signal_time": "2026-01-01 10:00", "entry_time": "2026-01-01 10:15",
             "exit_time": "2026-01-01 14:00", "net_return": 0.08, "holding_minutes": 225.0, "month": "2026-01"},
        ]
    )
    gated = pd.Series([True])
    baseline = _mode_metrics(pool, pd.Series([False]), "baseline", max_positions=8)
    half = _mode_metrics(pool, gated, "symbol_half_size", max_positions=8, half_size=True, half_size_factor=0.5)
    assert half["longs_gated"] == 1
    assert abs(half["portfolio_net20"] - 0.5 * baseline["portfolio_net20"]) < 1e-9
    assert half["portfolio_net20"] < baseline["portfolio_net20"]
