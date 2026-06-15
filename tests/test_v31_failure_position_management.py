from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v31_failure_position_management import (
    _apply_failure_position_rule,
    _attach_first_failure_event,
)


def test_failure_position_event_must_be_after_entry_and_before_planned_exit() -> None:
    base = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    sample = pd.DataFrame(
        [
            {
                "trade_key": "a",
                "symbol": "AAAUSDT",
                "entry_time": base + pd.Timedelta(hours=1),
                "effective_exit_time": base + pd.Timedelta(hours=4),
                "exit_time": base + pd.Timedelta(hours=4),
                "entry_price": 100.0,
                "cost_single_side_bps": 20.0,
                "effective_net_return": 0.05,
            },
            {
                "trade_key": "b",
                "symbol": "BBBUSDT",
                "entry_time": base + pd.Timedelta(hours=1),
                "effective_exit_time": base + pd.Timedelta(hours=2),
                "exit_time": base + pd.Timedelta(hours=2),
                "entry_price": 50.0,
                "cost_single_side_bps": 20.0,
                "effective_net_return": 0.02,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "motif": "S1", "feature_time": base + pd.Timedelta(minutes=45)},
            {"symbol": "AAAUSDT", "motif": "S3", "feature_time": base + pd.Timedelta(hours=2)},
            {"symbol": "BBBUSDT", "motif": "S5", "feature_time": base + pd.Timedelta(hours=3)},
        ]
    )
    prices = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "feature_time": base + pd.Timedelta(hours=2), "close": 99.0, "high": 101.0, "low": 98.0},
            {"symbol": "BBBUSDT", "feature_time": base + pd.Timedelta(hours=3), "close": 45.0, "high": 46.0, "low": 44.0},
        ]
    )
    attached = _attach_first_failure_event(sample, events, prices)
    assert bool(attached.loc[0, "failure_event_during_position"]) is True
    assert attached.loc[0, "failure_event_motif"] == "S3"
    assert bool(attached.loc[1, "failure_event_during_position"]) is False

    exited = _apply_failure_position_rule(attached, "exit_any", "exit_any")
    assert bool(exited.loc[0, "failure_position_exit"]) is True
    assert bool(exited.loc[1, "failure_position_exit"]) is False
    assert pd.Timestamp(exited.loc[0, "effective_exit_time"]) == base + pd.Timedelta(hours=2)
