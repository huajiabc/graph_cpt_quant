from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.reports.v103_graph_bucket_return_diffusion import (
    add_v103_states,
    build_v103_bucket_panel,
    build_v103_events,
    build_v103_portfolios,
)


def _synthetic_graph_frame() -> tuple[pd.DataFrame, dict[tuple[pd.Timestamp, str], list[str]]]:
    month = pd.Timestamp("2026-01-01", tz="UTC")
    times = pd.date_range("2026-01-01 00:15Z", periods=20, freq="15min")
    rows = []
    returns = {"A": 0.001, "B": 0.010, "C": 0.008, "D": 0.006}
    for symbol, ret_1h in returns.items():
        for index, timestamp in enumerate(times):
            rows.append(
                {
                    "symbol": symbol,
                    "feature_time": timestamp,
                    "entry_time": timestamp,
                    "month_start": month,
                    "ret_15m": 0.001 if symbol == "A" else 0.002,
                    "ret_1h": ret_1h,
                    "ret_4h": ret_1h * 2,
                    "future_ret_4h": 0.02 if symbol == "A" else 0.01,
                    "future_ret_12h": 0.03 if symbol == "A" else 0.015,
                }
            )
    mapping = {(month, "A"): ["B", "C", "D"]}
    return pd.DataFrame(rows), mapping


def test_bucket_panel_excludes_target_and_builds_catchup_return() -> None:
    frame, mapping = _synthetic_graph_frame()

    panel = build_v103_bucket_panel(frame, mapping, min_neighbors=3)
    row = panel.iloc[0]

    assert row["bucket_ret_1h"] == pytest.approx((0.010 + 0.008 + 0.006) / 3)
    assert row["target_lag_gap_1h"] == pytest.approx(0.007)
    assert row["catchup_gross_4h"] == pytest.approx(0.01)
    assert row["catchup_net_4h_40bp"] == pytest.approx(0.006)


def test_frozen_states_separate_turn_and_no_turn() -> None:
    base = pd.DataFrame(
        {
            "bucket_ret_1h": [0.01, 0.01, 0.01],
            "bucket_ret_1h_rank": [0.9, 0.9, 0.9],
            "bucket_positive_breadth_1h": [0.8, 0.8, 0.8],
            "bucket_excess_ret_1h": [0.005, 0.005, 0.005],
            "target_lag_gap_1h": [0.005, 0.005, 0.0005],
            "ret_15m": [0.001, -0.001, 0.001],
            "ret_1h": [0.004, 0.004, 0.012],
        }
    )

    states = add_v103_states(base)

    assert states["GBR1_BROAD_LAG_CATCHUP"].tolist() == [True, False, False]
    assert states["GBR2_LAG_NO_TURN"].tolist() == [False, True, False]
    assert states["GBR3_COIMPULSE_CONTINUATION"].tolist() == [False, False, True]


def test_transition_cooldown_and_portfolio_cap() -> None:
    frame, mapping = _synthetic_graph_frame()
    panel = build_v103_bucket_panel(frame, mapping, min_neighbors=3)
    panel["GBR1_BROAD_LAG_CATCHUP"] = True
    panel["GBR2_LAG_NO_TURN"] = False
    panel["GBR3_COIMPULSE_CONTINUATION"] = False

    events = build_v103_events(panel, cooldown_hours=4)
    portfolios, selected = build_v103_portfolios(events, max_positions=3)

    assert len(events) == 1
    assert len(portfolios) == 1
    assert len(selected) == 1
