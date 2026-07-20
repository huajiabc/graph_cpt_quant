from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    build_v132_july_membership,
)


def test_july_membership_is_balanced_and_excludes_btc() -> None:
    rng = np.random.default_rng(7)
    times = pd.date_range("2026-06-01", periods=120, freq="h", tz="UTC")
    rows = []
    for symbol in ["BTCUSDT", *[f"S{index:02d}" for index in range(16)]]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(times))))
        rows.extend(
            {"symbol": symbol, "feature_time": time, "close": value}
            for time, value in zip(times, close, strict=True)
        )
    membership = build_v132_july_membership(pd.DataFrame(rows), min_samples=100)
    assert membership["community_id"].nunique() == 8
    assert "BTCUSDT" not in set(membership["symbol"])
    assert membership["symbol"].nunique() == 16
