from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v163_within_bucket_liquidity_quality import (
    FROZEN_PAIRS,
    select_v163_pairs,
)


def test_select_v163_pairs_selects_one_member_each_side() -> None:
    rows = []
    for pair_index, pair in enumerate(FROZEN_PAIRS.values()):
        rows.append({"symbol": pair[0], "quality": float(pair_index + 1)})
        rows.append({"symbol": pair[1], "quality": float(-pair_index - 1)})
    local = pd.DataFrame(rows)
    longs, shorts = select_v163_pairs(local, "quality")
    assert len(longs) == len(shorts) == 8
    assert set(longs).isdisjoint(shorts)
    assert set(longs) | set(shorts) == set(local["symbol"])
