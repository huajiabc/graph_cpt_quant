from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v138_negative_funding_oi_state import (
    CANDIDATES,
    _eligible_oi_state,
)


def test_oi_state_branches_partition_negative_funding_names() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score_7d": [-0.2, -0.1, -0.3, 0.1],
            "oi_change_7d": [0.2, -0.1, 0.0, 0.3],
            "price_return": [0.0] * 4,
            "btc_beta": [1.0] * 4,
        }
    )
    build = _eligible_oi_state(frame, CANDIDATES[0])
    unwind = _eligible_oi_state(frame, CANDIDATES[1])
    assert build["symbol"].tolist() == ["A"]
    assert unwind["symbol"].tolist() == ["C", "B"]
