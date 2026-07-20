from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v139_short_crowding_pressure_rebound import (
    _ranked_pressure,
)


def test_pressure_rank_rewards_negative_funding_oi_build_and_taker_sell() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["HIGH", "LOW", "POSITIVE"],
            "score_7d": [-0.2, -0.1, 0.1],
            "oi_change_7d": [0.2, -0.1, 0.4],
            "taker_log_mean_7d": [-0.3, 0.2, -0.5],
            "price_return": [0.0] * 3,
            "btc_beta": [1.0] * 3,
        }
    )
    ranked = _ranked_pressure(frame)
    assert ranked["symbol"].tolist() == ["HIGH", "LOW"]
    assert ranked.iloc[0]["pressure_score"] > ranked.iloc[1]["pressure_score"]
