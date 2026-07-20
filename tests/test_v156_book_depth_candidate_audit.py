from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from pressure_graph.reports.v156_book_depth_candidate_audit import (
    independent_one_percent_median,
)


def test_independent_one_percent_median(tmp_path) -> None:
    path = tmp_path / "sample.zip"
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:01",
                "2026-01-01 00:00:31",
                "2026-01-01 00:00:31",
            ],
            "percentage": [-1, 1, -1, 1],
            "notional": [3, 1, 6, 2],
        }
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample.csv", frame.to_csv(index=False))
    assert np.isclose(independent_one_percent_median(path), 0.5)
