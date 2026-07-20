from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.v101_exact_flow_persistence import (
    _btc_return,
    attach_v101_btc,
)


def _btc_frame() -> pd.DataFrame:
    times = pd.date_range("2026-04-01 00:00Z", periods=300, freq="1min")
    prices = np.linspace(100.0, 110.0, len(times))
    return pd.DataFrame({"bar_open_time": times, "open": prices})


def test_btc_return_uses_same_entry_and_240m_exit() -> None:
    btc = _btc_frame()
    signal = pd.Timestamp("2026-04-01 00:15Z")
    expected = btc.iloc[255].open / btc.iloc[15].open - 1.0

    assert _btc_return(btc, signal) == pytest.approx(expected)


def test_attach_btc_subtracts_two_leg_cost() -> None:
    btc = _btc_frame()
    signal = pd.Timestamp("2026-04-01 00:15Z")
    panel = pd.DataFrame(
        {
            "signal_time": [signal],
            "gross_return_240m": [0.10],
        }
    )

    attributed = attach_v101_btc(panel, btc)
    expected_relative = 0.10 - _btc_return(btc, signal)

    assert attributed.loc[0, "relative_gross_return_240m"] == pytest.approx(
        expected_relative
    )
    assert attributed.loc[0, "hedged_net_return_240m_40bp"] == pytest.approx(
        expected_relative - 0.004
    )
