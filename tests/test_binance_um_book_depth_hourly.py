from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from pressure_graph.binance_um_book_depth_hourly import (
    parse_hourly_book_depth_archive,
)


def test_hourly_archive_uses_strict_prior_hour(tmp_path) -> None:
    path = tmp_path / "sample.zip"
    rows = []
    for timestamp, scale in (
        ("2026-01-01 00:00:00", 1.0),
        ("2026-01-01 00:59:59", 2.0),
        ("2026-01-01 01:00:00", 3.0),
    ):
        for band in (1.0, 5.0):
            rows.extend(
                [
                    {
                        "timestamp": timestamp,
                        "percentage": -band,
                        "notional": 3.0 * scale,
                    },
                    {
                        "timestamp": timestamp,
                        "percentage": band,
                        "notional": 1.0 * scale,
                    },
                ]
            )
    frame = pd.DataFrame(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample.csv", frame.to_csv(index=False))
    hourly = parse_hourly_book_depth_archive(path)
    assert hourly["decision_time"].tolist() == [
        pd.Timestamp("2026-01-01 01:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 02:00:00", tz="UTC"),
    ]
    assert hourly["notional_imbalance_1p0_valid_snapshots"].tolist() == [2, 1]
    assert np.isclose(hourly["notional_imbalance_1p0_median"], 0.5).all()
    assert hourly["total_notional_1p0_median"].tolist() == [6.0, 12.0]
