from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v126_turnover_governed_cross_venue_carry import (
    _select_hold_band,
)


def test_hold_band_retains_rank_ten_before_filling_from_top() -> None:
    symbols = [f"S{index:02d}" for index in range(20)]
    frame = pd.DataFrame(
        {
            "symbol": symbols,
            "score_30d": list(reversed(range(1, 21))),
            "pair_gross_return": [0.0] * 20,
        }
    )
    previous = ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S09"]
    selected = _select_hold_band(frame, previous, bucket_size=9, hold_rank=18)
    assert "S09" in selected
    assert len(selected) == 9


def test_negative_or_below_hold_rank_name_is_replaced() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score_30d": [0.4, 0.3, 0.2, -0.1],
            "pair_gross_return": [0.0] * 4,
        }
    )
    selected = _select_hold_band(
        frame, ["D", "C"], bucket_size=2, hold_rank=3
    )
    assert selected == ["C", "A"]


def test_full_retained_book_never_expands_past_bucket_size() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["NEW", "A", "B", "C"],
            "score_30d": [0.5, 0.4, 0.3, 0.2],
            "pair_gross_return": [0.0] * 4,
        }
    )
    selected = _select_hold_band(frame, ["A", "B"], bucket_size=2, hold_rank=4)
    assert selected == ["A", "B"]
