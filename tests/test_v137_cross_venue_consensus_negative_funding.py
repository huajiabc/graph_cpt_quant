from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v137_cross_venue_consensus_negative_funding import (
    V137Config,
    _select_dual_negative_hold_band,
)


def test_dual_negative_selector_rejects_single_venue_negative() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "score_7d": [-0.1] * 6,
            "binance_score_7d": [-0.1, -0.2, -0.3, -0.4, 0.1, 0.2],
            "consensus_score_7d": [-0.2, -0.3, -0.4, -0.5, 0.0, 0.1],
            "price_return": [0.0] * 6,
            "btc_beta": [1.0] * 6,
        }
    )
    selected = _select_dual_negative_hold_band(frame, [], V137Config())
    assert selected == ["D", "C", "B", "A"]
