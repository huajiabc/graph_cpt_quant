from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from pressure_graph.reports.v160_hourly_depth_candidate_audit import (
    independent_hourly_median,
)


def test_independent_hourly_median_is_half_open(tmp_path) -> None:
    path = tmp_path / "sample.zip"
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 11:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 11:59:59",
                "2026-01-01 11:59:59",
                "2026-01-01 12:00:00",
                "2026-01-01 12:00:00",
            ],
            "percentage": [-1, 1, -1, 1, -1, 1],
            "notional": [3, 1, 6, 2, 1, 3],
        }
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample.csv", frame.to_csv(index=False))
    value, count = independent_hourly_median(
        path,
        pd.Timestamp("2026-01-01 11:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
    )
    assert count == 2
    assert np.isclose(value, 0.5)
