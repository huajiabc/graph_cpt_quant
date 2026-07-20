from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v176_monthly_deribit_surface_extension import (
    select_nearest_30d_surface,
)


def test_select_nearest_30d_surface_breaks_tie_to_earlier_expiry() -> None:
    date = pd.Timestamp("2024-01-10", tz="UTC")
    surface = pd.DataFrame(
        {
            "quality_pass": [True, True, True],
            "feature_time": [date, date, date + pd.Timedelta(days=1)],
            "expiration_time": [
                pd.Timestamp("2024-02-08 08:00", tz="UTC"),
                pd.Timestamp("2024-02-12 08:00", tz="UTC"),
                pd.Timestamp("2024-02-12 08:00", tz="UTC"),
            ],
            "dte": [29.0, 31.0, 30.0],
        }
    )
    selected = select_nearest_30d_surface(surface)
    assert len(selected) == 2
    assert selected.iloc[0]["expiration_time"] == pd.Timestamp(
        "2024-02-08 08:00", tz="UTC"
    )
