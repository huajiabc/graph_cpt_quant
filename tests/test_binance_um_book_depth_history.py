from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd

from pressure_graph.binance_um_book_depth_history import (
    book_depth_daily_url,
    parse_book_depth_zip,
)


def _fixture_zip(frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("TEST-bookDepth-2026-01-01.csv", frame.to_csv(index=False))
    return output.getvalue()


def test_book_depth_daily_url() -> None:
    assert book_depth_daily_url("SOLUSDT", date(2026, 1, 2)).endswith(
        "/SOLUSDT/SOLUSDT-bookDepth-2026-01-02.zip"
    )


def test_parse_book_depth_zip_uses_signed_pairs() -> None:
    rows = []
    for timestamp, multiplier in (
        ("2026-01-01 00:00:01", 1.0),
        ("2026-01-01 00:00:31", 2.0),
    ):
        for band in (0.2, 1.0, 5.0):
            rows.extend(
                [
                    {
                        "timestamp": timestamp,
                        "percentage": -band,
                        "depth": 3.0 * multiplier,
                        "notional": 6.0 * multiplier,
                    },
                    {
                        "timestamp": timestamp,
                        "percentage": band,
                        "depth": 1.0 * multiplier,
                        "notional": 2.0 * multiplier,
                    },
                ]
            )
    content = _fixture_zip(pd.DataFrame(rows))
    result = parse_book_depth_zip(content, "TESTUSDT", date(2026, 1, 1))
    assert result["snapshot_count"] == 2
    for band in ("0p2", "1p0", "5p0"):
        assert np.isclose(result[f"notional_imbalance_{band}_median"], 0.5)
        assert np.isclose(result[f"depth_imbalance_{band}_mean"], 0.5)
        assert result[f"notional_imbalance_{band}_valid_snapshots"] == 2
