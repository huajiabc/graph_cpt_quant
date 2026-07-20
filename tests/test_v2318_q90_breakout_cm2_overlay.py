import numpy as np
import pandas as pd

from pressure_graph.reports.v2318_q90_breakout_cm2_overlay import (
    V2318Config,
    build_v2318_panel,
)


def test_v2318_compounds_events_in_realization_week() -> None:
    times = pd.date_range("2026-01-05", periods=3, freq="7D", tz="UTC")
    mapping = pd.DataFrame(
        {
            "event_entry_time": [times[0] + pd.Timedelta(hours=4), times[0] + pd.Timedelta(hours=8)],
            "portfolio_entry_time": [times[0], times[0]],
        }
    )
    outcomes = pd.DataFrame(
        {
            "entry_time": mapping["event_entry_time"],
            "triggered": [True, True],
            "ambiguous_trigger": [False, False],
            "primary_net_return": [0.01, 0.02],
            "stress_net_return": [0.009, 0.019],
            "reversed_primary_net_return": [-0.01, -0.02],
        }
    )
    core = pd.DataFrame(
        {
            "entry_time": times,
            "exit_time": times + pd.Timedelta(days=7),
            "month_start": pd.Timestamp("2026-01-01", tz="UTC"),
            "period": ["development", "validation", "holdout"],
            "primary_net_return": [0.03, 0.0, -0.01],
            "stress_net_return": [0.02, -0.01, -0.02],
        }
    )
    event_outcomes, panel = build_v2318_panel(
        mapping, outcomes, core, V2318Config(overlay_weight=0.10)
    )
    expected_satellite = (1.01 * 1.02) - 1.0
    assert len(event_outcomes) == 2
    assert np.isclose(panel.loc[0, "satellite_primary_return"], expected_satellite)
    assert np.isclose(
        panel.loc[0, "combined_primary_return"], 0.03 + 0.10 * expected_satellite
    )
    assert panel.loc[1:, "satellite_primary_return"].eq(0).all()
