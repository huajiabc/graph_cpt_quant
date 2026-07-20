from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v161_hourly_liquidity_withdrawal_amplification import (
    add_v161_scores,
)


def test_add_v161_scores_uses_withdrawal_rank_without_flipping_direction() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2026-01-01", tz="UTC")] * 3,
            "prior_residual_return": [0.01, -0.02, 0.03],
            "total_depth_1pct": [50.0, 100.0, 80.0],
            "previous_total_depth_1pct": [100.0, 100.0, 100.0],
            "total_depth_5pct": [50.0, 100.0, 80.0],
            "previous_total_depth_5pct": [100.0, 100.0, 100.0],
        }
    )
    out = add_v161_scores(frame)
    assert out["withdrawal_percentile_1pct"].tolist() == [1.0, 1 / 3, 2 / 3]
    assert np.sign(out["score_1pct"]).tolist() == [1.0, -1.0, 1.0]
