from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.backtest.short_execution import (
    ShortExitRule,
    short_net_return,
    simulate_short_exit,
)
from pressure_graph.labels.short import add_short_labels, squeeze_before_hit


def _group(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_squeeze_before_hit_flags_up_move_first():
    close = pd.Series([100.0, 100.0, 100.0, 100.0])
    high = pd.Series([100.0, 103.0, 100.0, 100.0])  # +3% on bar 1
    low = pd.Series([100.0, 100.0, 96.0, 96.0])  # -4% on bar 2
    result = squeeze_before_hit(close, high, low, window=3, down_target=0.03, squeeze_threshold=0.02)
    assert bool(result.iloc[0]) is True


def test_squeeze_before_hit_clean_drop_no_squeeze():
    close = pd.Series([100.0, 100.0, 100.0])
    high = pd.Series([100.0, 100.5, 100.0])  # never +2%
    low = pd.Series([100.0, 96.0, 95.0])  # -4% straight down
    result = squeeze_before_hit(close, high, low, window=2, down_target=0.03, squeeze_threshold=0.02)
    assert bool(result.iloc[0]) is False


def test_add_short_labels_columns_and_signs():
    times = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC")
    close = np.linspace(100, 80, 40)  # steady decline -> short-favorable
    frame = pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": "AAAUSDT",
            "bar_open_time": times,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        }
    )
    labeled = add_short_labels(frame, {"4h": 16})
    assert "future_max_down_4h" in labeled.columns
    assert "hit_down_3pct_4h" in labeled.columns
    assert "up_2pct_before_down_3pct_4h" in labeled.columns
    # In a steady decline the early bars must register downside capture.
    assert bool(labeled["hit_down_3pct_4h"].iloc[0])
    assert labeled["future_max_down_4h"].iloc[0] < 0


def test_simulate_short_exit_take_profit():
    group = _group([(100, 100, 99, 99.5), (99, 99.5, 97, 98), (98, 98.5, 95, 96)])
    exit = simulate_short_exit(group, 0, 100.0, ShortExitRule(0.03, 0.025, 3))
    assert exit.exit_reason == "take_profit"
    assert exit.gross_return > 0
    assert exit.squeezed is False


def test_simulate_short_exit_stop_is_a_squeeze():
    group = _group([(100, 103, 100, 102), (102, 104, 101, 103)])
    exit = simulate_short_exit(group, 0, 100.0, ShortExitRule(0.03, 0.025, 2))
    assert exit.exit_reason == "stop"
    assert exit.gross_return < 0  # short loses when price rallies
    assert exit.squeezed is True
    assert exit.max_adverse_excursion >= 0.025


def test_simulate_short_exit_ambiguous_resolves_stop_first():
    # One bar touches both the +2.5% stop and the -3% target.
    group = _group([(100, 103, 96, 100)])
    exit = simulate_short_exit(group, 0, 100.0, ShortExitRule(0.03, 0.025, 1))
    assert exit.exit_reason == "stop_ambiguous"
    assert exit.squeezed is True


def test_short_net_return_applies_extra_slippage():
    gross = 0.03
    net = short_net_return(gross, cost_single_side_bps=20.0, extra_slippage_bps=5.0)
    # round trip = 2 * (20 + 5) bp = 50bp = 0.005
    assert abs(net - (gross - 0.005)) < 1e-12
