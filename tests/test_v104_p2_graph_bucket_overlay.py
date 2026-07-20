from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v104_p2_graph_bucket_overlay import add_v104_overlay_flags


def test_v104_overlay_blocks_only_covered_strong_bucket_laggard() -> None:
    pool = pd.DataFrame(
        {
            "graph_covered": [True, True, False],
            "bucket_ret_1h": [0.01, 0.01, None],
            "bucket_ret_1h_rank": [0.9, 0.9, None],
            "bucket_positive_breadth_1h": [0.8, 0.8, None],
            "bucket_excess_ret_1h": [0.004, 0.004, None],
            "target_lag_gap_1h": [0.005, 0.001, None],
        }
    )

    out = add_v104_overlay_flags(pool)

    assert out["strong_bucket_laggard"].tolist() == [True, False, False]
    assert out["overlay_keep"].tolist() == [False, True, True]
