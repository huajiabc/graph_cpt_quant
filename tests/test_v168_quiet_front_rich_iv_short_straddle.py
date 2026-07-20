from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v168_quiet_front_rich_iv_short_straddle import (
    _tail_metrics,
    evaluate_v168,
)


def test_tail_metrics_use_worst_five_percent() -> None:
    values = pd.Series([-0.04, -0.02, *([0.01] * 38)])
    worst, shortfall = _tail_metrics(values)
    assert worst == -0.04
    assert shortfall == -0.03


def test_v168_signal_requires_rich_iv_and_quiet_front() -> None:
    times = pd.date_range("2023-07-01 01:00:00Z", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "entry_time": times,
            "period": ["development"] * 8,
            "iv_rv_spread": [0.11, 0.09, 0.12, 0.15, 0.20, 0.11, 0.11, 0.11],
            "front_gap": [-0.1, -0.1, 0.1, -0.1, -0.2, -0.1, -0.1, -0.1],
            "alt_high_vol_breadth": [0.1, 0.1, 0.1, 0.5, 0.2, 0.1, 0.1, 0.1],
            "short_gross_return": [0.01] * 8,
            "short_primary_net_return": [0.008] * 8,
            "short_stress_net_return": [0.006] * 8,
            "primary_net_return": [-0.008] * 8,
        }
    )
    results = evaluate_v168(frame, circular_draws=20, circular_seed=1)
    selected = results["candidate"]
    assert selected["entry_time"].tolist() == [times[0], times[4], times[5], times[6], times[7]]
