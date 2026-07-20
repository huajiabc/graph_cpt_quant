from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    V133Config,
    _moving_block_means,
    _random_hold_band,
    aggregate_v133_weeks,
)


def test_random_hold_band_is_full_positive_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "symbol": [f"S{index:02d}" for index in range(25)],
            "score_30d": [1.0] * 20 + [-1.0] * 5,
            "pair_gross_return": [0.0] * 25,
        }
    )
    cfg = V133Config(bucket_size=9, hold_rank=18)
    first = _random_hold_band(frame, [], cfg, np.random.default_rng(7))
    second = _random_hold_band(frame, [], cfg, np.random.default_rng(7))
    assert first == second
    assert len(first) == 9
    assert all(int(symbol[1:]) < 20 for symbol in first)


def test_week_aggregation_excludes_burn_in_and_charges_round_trip() -> None:
    times = pd.date_range("2025-08-04", periods=21, freq="D", tz="UTC")
    daily = pd.DataFrame(
        {
            "entry_time": times,
            "exit_time": times + pd.Timedelta(days=1),
            "active_cohorts": [min(index + 1, 7) for index in range(21)],
            "coverage": [70] * 21,
            "price_basis_return": [0.0] * 21,
            "funding_spread_return": [0.001] * 21,
            "gross_return": [0.001] * 21,
            "portfolio_turnover": [1.0 / 7.0] * 7 + [0.0] * 14,
        }
    )
    weekly = aggregate_v133_weeks(daily)
    assert len(weekly) == 2
    assert np.isclose(weekly.iloc[0]["realized_turnover"], 1.0)
    assert np.isclose(weekly.iloc[-1]["realized_turnover"], 1.0)
    assert np.allclose(weekly["gross_return"], 0.007)


def test_moving_block_bootstrap_constant_series_is_constant() -> None:
    draws = _moving_block_means(np.full(12, 0.003), 20, 4, np.random.default_rng(3))
    assert np.allclose(draws, 0.003)
