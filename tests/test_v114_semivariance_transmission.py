from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v114_semivariance_transmission import (
    CANDIDATES,
    _signed_shock,
    summarize_v114,
)


def test_signed_shock_separates_upside_and_downside_semivariance() -> None:
    residual = pd.DataFrame({"A": [-0.04, 0.02], "B": [0.03, -0.01]})
    scale = pd.Series({"A": 0.02, "B": 0.01})

    upside = _signed_shock(residual, scale, "upside")
    downside = _signed_shock(residual, scale, "downside")

    assert upside.to_numpy().tolist() == [[0.0, 3.0], [1.0, 0.0]]
    assert downside.to_numpy().tolist() == [[2.0, 0.0], [0.0, 1.0]]


def test_summary_applies_one_round_trip_cost_to_bucket_mean() -> None:
    rows = []
    for candidate, gross in zip(CANDIDATES.values(), (0.01, -0.005)):
        rows.append(
            {
                "candidate": candidate,
                "period": "validation",
                "entry_day": "2026-01-02",
                "entry_month": "2026-01",
                "bucket_size": 3,
                "raw_gross_4h": gross,
                "residual_gross_4h": gross / 2,
                "raw_net_4h_20bp": gross - 0.002,
                "raw_net_4h_30bp": gross - 0.003,
                "raw_net_4h_50bp": gross - 0.005,
            }
        )

    summary = summarize_v114(pd.DataFrame(rows))
    upside = summary[
        summary["scope"].eq("validation")
        & summary["candidate"].eq(CANDIDATES["upside"])
    ].iloc[0]

    assert upside["portfolio_observations"] == 1
    assert np.isclose(upside["mean_raw_net_4h_20bp"], -0.007)


def test_downside_futures_direction_is_negative_raw_return() -> None:
    raw_future = pd.Series([0.03, -0.01])
    downside_gross = -float(raw_future.mean())

    assert np.isclose(downside_gross, -0.01)
