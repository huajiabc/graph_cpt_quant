from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v134_negative_funding_beta_neutral_rebound import (
    V134Config,
    _select_negative_hold_band,
    _weights_and_components,
    build_v134_portfolio,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [f"S{index:02d}" for index in range(20)],
            "score_7d": np.linspace(-0.02, -0.001, 20),
            "btc_beta": [1.5] * 20,
            "price_return": [0.01] * 20,
            "future_funding": [-0.002] * 20,
            "btc_return": [0.004] * 20,
            "btc_future_funding": [0.0007] * 20,
        }
    )


def test_negative_hold_band_retains_eligible_name() -> None:
    cfg = V134Config(bucket_size=3, hold_rank=6)
    selected = _select_negative_hold_band(_frame(), ["S04", "S10"], cfg)
    assert selected == ["S04", "S00", "S01"]


def test_beta_neutral_weights_have_unit_gross_and_zero_beta() -> None:
    cfg = V134Config(bucket_size=3)
    weights, components = _weights_and_components(_frame(), ["S00", "S01", "S02"], cfg)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    assert np.isclose(components["residual_btc_beta"], 0.0)
    assert components["funding_return"] > 0


def test_cash_gap_charges_exit_and_reentry() -> None:
    cfg = V134Config(bucket_size=2, hold_rank=4)
    frames = []
    for week, negative_count in enumerate((3, 1, 3)):
        frame = _frame().iloc[:3].copy()
        frame["entry_time"] = pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(days=7 * week)
        frame["exit_time"] = frame["entry_time"] + pd.Timedelta(days=7)
        frame["month_start"] = pd.Timestamp("2026-01-01", tz="UTC")
        frame["period"] = "validation"
        frame.loc[frame.index[negative_count:], "score_7d"] = 0.001
        frames.append(frame)
    portfolio = build_v134_portfolio(pd.concat(frames, ignore_index=True), cfg)
    assert len(portfolio) == 2
    assert np.isclose(portfolio.iloc[0]["realized_turnover"], 2.0)
    assert np.isclose(portfolio.iloc[1]["realized_turnover"], 2.0)
