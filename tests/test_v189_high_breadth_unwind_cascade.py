from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v189_high_breadth_unwind_cascade import (
    CANDIDATE,
    V189Config,
    build_v189_events,
    random_v189_controls,
)


def test_cascade_event_continues_source_direction_for_one_bar() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": [100, 101, 102]}, index=index)
    signals = pd.DataFrame(
        {
            "feature_time": [index[0]],
            "source_feature_time": [index[0]],
            "kind": ["unwind"],
            "candidate": ["old"],
            "source_sign": [1.0],
            "volatility_breadth": [0.8],
        }
    )
    events = build_v189_events(signals, close, V189Config())
    assert events.loc[0, "candidate"] == CANDIDATE
    assert events.loc[0, "trade_direction"] == 1.0
    assert math.isclose(events.loc[0, "gross_return"], 0.01)
    assert math.isclose(events.loc[0, "primary_net_return"], 0.009)


def test_random_controls_preserve_month_counts_and_are_deterministic() -> None:
    selected = pd.DataFrame(
        {
            "entry_month": ["2026-01", "2026-01"],
            "primary_net_return": [0.01, -0.01],
        }
    )
    base = pd.DataFrame(
        {
            "entry_month": ["2026-01"] * 4,
            "primary_net_return": [0.01, -0.01, 0.02, -0.02],
        }
    )
    cfg = V189Config(random_iterations=5, seed=189)
    first = random_v189_controls(selected, base, cfg)
    second = random_v189_controls(selected, base, cfg)
    pd.testing.assert_frame_equal(first, second)
    assert first["events"].eq(2).all()
