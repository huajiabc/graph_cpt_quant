from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v32_failure_state_atlas import V32Config, _best_action, _future_long_labels


def test_future_long_label_marks_negative_window_as_no_long() -> None:
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    events = pd.DataFrame(
        [{"failure_event_id": "e1", "symbol": "AAAUSDT", "motif": "S1", "feature_time": ts}]
    )
    sample = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "signal_time": ts + pd.Timedelta(minutes=15), "net_return_at_cost": -0.02},
            {"symbol": "AAAUSDT", "signal_time": ts + pd.Timedelta(minutes=30), "net_return_at_cost": 0.00},
            {"symbol": "AAAUSDT", "signal_time": ts + pd.Timedelta(hours=13), "net_return_at_cost": 0.10},
        ]
    )
    labels = _future_long_labels(events, sample, V32Config())
    row = labels.iloc[0]
    assert int(row["future_long_count_48"]) == 2
    assert row["future_long_label"] == "no_long_48_best"


def test_best_action_keeps_existing_when_exit_counterfactual_is_bad() -> None:
    row = pd.Series({"future_long_label": "no_long_48_best", "position_action_label": "keep_long_best"})
    assert _best_action(row) == "keep_existing_but_no_new_long_48"
