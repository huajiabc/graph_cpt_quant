from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from pressure_graph.reports.v162_liquidity_withdrawal_candidate_audit import (
    independent_total_depth_median,
)


def test_independent_total_depth_median(tmp_path) -> None:
    path = tmp_path / "sample.zip"
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 11:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 11:30:00",
                "2026-01-01 11:30:00",
            ],
            "percentage": [-1, 1, -1, 1],
            "notional": [3, 1, 6, 2],
        }
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample.csv", frame.to_csv(index=False))
    value, count = independent_total_depth_median(
        path,
        pd.Timestamp("2026-01-01 11:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
    )
    assert count == 2
    assert np.isclose(value, 6.0)
