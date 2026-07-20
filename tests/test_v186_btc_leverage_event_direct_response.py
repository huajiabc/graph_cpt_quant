from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v185_btc_leverage_flow_graph import BUILD, UNWIND
from pressure_graph.reports.v186_btc_leverage_event_direct_response import (
    BUILD_CANDIDATE,
    UNWIND_CANDIDATE,
    V186Config,
    build_v186_events,
    random_v186_circular_controls,
)


def test_direct_event_directions_and_returns_are_frozen() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": [100, 101, 102, 103, 104]}, index=index)
    signals = pd.DataFrame(
        {
            "feature_time": [index[0], index[1]],
            "source_feature_time": [index[0], index[1]],
            "kind": [BUILD, UNWIND],
            "candidate": ["old_build", "old_unwind"],
            "source_sign": [1.0, 1.0],
        }
    )
    events = build_v186_events(signals, close, V186Config(holding_bars=2))
    build = events.set_index("candidate").loc[BUILD_CANDIDATE]
    unwind = events.set_index("candidate").loc[UNWIND_CANDIDATE]
    assert build["trade_direction"] == 1.0
    assert unwind["trade_direction"] == -1.0
    assert math.isclose(build["gross_return"], 0.02)
    assert math.isclose(unwind["gross_return"], -(103 / 101 - 1))
    assert math.isclose(
        build["primary_net_return"], build["gross_return"] - 0.001
    )


def test_circular_controls_are_deterministic_and_preserve_event_counts() -> None:
    index = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC")
    close = pd.DataFrame(
        {"BTCUSDT": [100 + position for position in range(len(index))]},
        index=index,
    )
    events = pd.DataFrame(
        {
            "candidate": [BUILD_CANDIDATE] * 2 + [UNWIND_CANDIDATE] * 2,
            "entry_month": ["2026-01"] * 4,
            "entry_time": [index[2], index[8], index[4], index[10]],
            "trade_direction": [1.0, -1.0, -1.0, 1.0],
        }
    )
    cfg = V186Config(random_iterations=5, holding_bars=2, seed=186)
    first = random_v186_circular_controls(events, close, cfg)
    second = random_v186_circular_controls(events, close, cfg)
    pd.testing.assert_frame_equal(first, second)
    candidate_rows = first[first["candidate"].ne("FAMILY_MAX")]
    assert candidate_rows["events"].eq(2).all()
    assert len(first) == 15
