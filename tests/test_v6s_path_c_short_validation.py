from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v6s_path_c_short_validation import (
    CANDIDATE_SC1,
    CANDIDATE_SC2,
    CANDIDATE_SC3A,
    CANDIDATE_SC3B,
    CANDIDATES,
    V6SConfig,
    _abcd_table,
    _candidate_outcomes_for_event,
    _candidate_summary,
    _cost_stress_table,
    _forward_long_match,
    _hedge_table,
    _label_clean_short,
    _long_stack_monthly_dd,
    _net_short_return,
    _stability_table,
)


def _make_event_row(month: str = "2026-01", symbol: str = "AAAUSDT", gross: float = 0.02, hold: int = 48) -> dict:
    return {
        "symbol": symbol,
        "signal_time": pd.Timestamp("2026-01-15 10:00", tz="UTC"),
        "entry_time": pd.Timestamp("2026-01-15 10:15", tz="UTC"),
        "month": month,
        "execution": "swing",
        "gross_return": gross,
        "holding_bars": hold,
        "squeezed": False,
        "max_adverse_excursion": 0.005,
        "max_favorable_excursion": 0.03,
        "exit_reason": "take_profit",
        "confirmation_idx": 5,
    }


def test_four_candidates_defined() -> None:
    assert CANDIDATES == (
        CANDIDATE_SC1, CANDIDATE_SC2, CANDIDATE_SC3A, CANDIDATE_SC3B,
    )


def test_net_short_return_applies_cost_and_funding() -> None:
    cfg = V6SConfig()
    # gross 2%, 48 bars hold (12h), 20bp focal cost, funding APR 30%
    # net = 0.02 - 2*20/10000 + 0.30 * 12 / (24*365)
    net = _net_short_return(0.02, 48, 20.0, 0.0, cfg)
    expected = 0.02 - 0.004 + 0.30 * 12 / (24 * 365)
    assert abs(net - expected) < 1e-9


def test_net_short_return_extra_slippage_doubles_cost_band() -> None:
    cfg = V6SConfig()
    base = _net_short_return(0.02, 16, 20.0, 0.0, cfg)
    with_slip = _net_short_return(0.02, 16, 20.0, 10.0, cfg)
    assert base - with_slip > 0
    assert abs((base - with_slip) - 2 * 10 / 10_000) < 1e-9


def test_label_clean_short_records_drop_and_no_squeeze() -> None:
    # Synthetic: price drops steadily, never rallies above entry.
    df = pd.DataFrame({
        "open": [100.0] * 20,
        "high": np.linspace(100, 96, 20),
        "low": np.linspace(99, 90, 20),
        "close": np.linspace(99.5, 91, 20),
    })
    labels = _label_clean_short(df, entry_idx=0, cfg=V6SConfig())
    assert labels["hit_down_3pct"] is True
    assert labels["hit_down_5pct"] is True
    assert labels["up_before_down_2pct"] is False
    assert labels["short_squeeze_before_hit"] is False


def test_label_clean_short_flags_squeeze_then_resolution() -> None:
    # Synthetic: bar 1 rallies 3% then bars 2..10 drop 5%.
    n = 12
    high = [100.0] + [103.0] + list(np.linspace(101, 96, n - 2))
    low = [99.5] + [102.0] + list(np.linspace(99, 90, n - 2))
    close = [100.0] + [102.5] + list(np.linspace(100, 92, n - 2))
    df = pd.DataFrame({"open": [100.0] * n, "high": high, "low": low, "close": close})
    labels = _label_clean_short(df, entry_idx=0, cfg=V6SConfig())
    assert labels["up_before_down_2pct"] is True
    assert labels["hit_down_3pct"] is True
    assert labels["short_squeeze_before_hit"] is True


def test_forward_long_match_only_returns_window_entries() -> None:
    by_sym = {
        "AAAUSDT": [
            (int(pd.Timestamp("2026-01-15 11:00", tz="UTC").value), {"net_return": -0.005, "entry_time": pd.Timestamp("2026-01-15 11:00", tz="UTC"), "exit_time": pd.Timestamp("2026-01-15 15:00", tz="UTC")}),
            (int(pd.Timestamp("2026-01-15 23:00", tz="UTC").value), {"net_return": 0.01, "entry_time": pd.Timestamp("2026-01-15 23:00", tz="UTC"), "exit_time": pd.Timestamp("2026-01-16 03:00", tz="UTC")}),
        ]
    }
    inside = _forward_long_match("AAAUSDT", pd.Timestamp("2026-01-15 10:00", tz="UTC"), by_sym, window_minutes=720)
    assert len(inside) == 1  # only the 11:00 long (the 23:00 long is 13h later → outside 12h window)
    assert inside[0]["net_return"] == -0.005


def test_candidate_outcomes_a_equals_minus_b() -> None:
    event = _make_event_row()
    by_sym = {
        "AAAUSDT": [
            (int(pd.Timestamp("2026-01-15 11:00", tz="UTC").value), {"net_return": -0.008, "entry_time": pd.Timestamp("2026-01-15 11:00", tz="UTC"), "exit_time": pd.Timestamp("2026-01-15 15:00", tz="UTC")}),
        ]
    }
    out = _candidate_outcomes_for_event(event, by_sym, V6SConfig())
    assert abs(out["A_no_action"] + out["B_no_long"]) < 1e-12
    assert out[CANDIDATE_SC3A] == out["B_no_long"]


def test_candidate_outcomes_short_size_scales_correctly() -> None:
    event = _make_event_row(gross=0.025, hold=32)
    out = _candidate_outcomes_for_event(event, {}, V6SConfig())
    sc1 = out[CANDIDATE_SC1]
    sc2 = out[CANDIDATE_SC2]
    assert abs(sc2 / sc1 - 0.5) < 1e-9


def test_cost_stress_table_produces_grid() -> None:
    events = pd.DataFrame([
        _make_event_row(gross=0.02, hold=48),
        _make_event_row(gross=-0.01, hold=24, month="2026-02"),
    ])
    table = _cost_stress_table(events, V6SConfig())
    # 4 cost levels × 3 slippage levels × 2 candidates = 24 rows
    assert len(table) == 24
    assert set(table["candidate"]) == {CANDIDATE_SC1, CANDIDATE_SC2}


def test_stability_table_reports_concentration() -> None:
    outcomes = pd.DataFrame([
        {"symbol": "AAA", "month": "2026-01", CANDIDATE_SC1: 0.05, CANDIDATE_SC2: 0.025, CANDIDATE_SC3A: 0.0, CANDIDATE_SC3B: 0.0},
        {"symbol": "AAA", "month": "2026-02", CANDIDATE_SC1: -0.01, CANDIDATE_SC2: -0.005, CANDIDATE_SC3A: 0.0, CANDIDATE_SC3B: 0.0},
        {"symbol": "BBB", "month": "2026-01", CANDIDATE_SC1: 0.02, CANDIDATE_SC2: 0.01, CANDIDATE_SC3A: 0.0, CANDIDATE_SC3B: 0.0},
    ])
    table = _stability_table(outcomes, V6SConfig())
    sc1_row = table[table["candidate"].eq(CANDIDATE_SC1)].iloc[0]
    # 2026-01 contributes (0.05 + 0.02) / total = 0.07 / 0.06 = 1.17 -> max share
    assert sc1_row["best_month"] == "2026-01"
    assert sc1_row["unique_months"] == 2


def test_hedge_table_correlates_candidate_vs_long() -> None:
    outcomes = pd.DataFrame([
        {"symbol": "AAA", "month": "2026-01", CANDIDATE_SC1: 0.05, CANDIDATE_SC2: 0.025, CANDIDATE_SC3A: 0.0, CANDIDATE_SC3B: 0.0},
        {"symbol": "AAA", "month": "2026-02", CANDIDATE_SC1: -0.02, CANDIDATE_SC2: -0.01, CANDIDATE_SC3A: 0.0, CANDIDATE_SC3B: 0.0},
        {"symbol": "AAA", "month": "2026-03", CANDIDATE_SC1: 0.03, CANDIDATE_SC2: 0.015, CANDIDATE_SC3A: 0.0, CANDIDATE_SC3B: 0.0},
    ])
    monthly_long = pd.Series({"2026-01": -0.04, "2026-02": 0.05, "2026-03": -0.03})
    table = _hedge_table(outcomes, monthly_long)
    # The candidate is anti-correlated with long stack month returns by construction
    sc1_row = table[table["candidate"].eq(CANDIDATE_SC1)].iloc[0]
    assert sc1_row["pearson_corr"] < 0


def test_abcd_table_flags_when_short_beats_no_long() -> None:
    outcomes = pd.DataFrame([
        {
            "symbol": "AAA",
            "signal_time": pd.Timestamp("2026-01-15", tz="UTC"),
            "month": "2026-01",
            "A_no_action": 0.01,
            "B_no_long": -0.01,
            "C_normal_short": 0.02,
            "D_small_short": 0.01,
        },
        {
            "symbol": "BBB",
            "signal_time": pd.Timestamp("2026-01-16", tz="UTC"),
            "month": "2026-01",
            "A_no_action": -0.02,
            "B_no_long": 0.02,
            "C_normal_short": 0.005,
            "D_small_short": 0.0025,
        },
    ])
    table = _abcd_table(outcomes)
    assert table.iloc[0]["C_beats_B"] is np.True_ or table.iloc[0]["C_beats_B"] is True
    assert table.iloc[1]["C_beats_B"] is np.False_ or table.iloc[1]["C_beats_B"] is False


def test_long_stack_monthly_dd_aggregates_per_month() -> None:
    trades = pd.DataFrame([
        {"symbol": "X", "entry_time": pd.Timestamp("2026-01-05", tz="UTC"), "net_return": 0.01},
        {"symbol": "X", "entry_time": pd.Timestamp("2026-01-20", tz="UTC"), "net_return": -0.03},
        {"symbol": "Y", "entry_time": pd.Timestamp("2026-02-01", tz="UTC"), "net_return": 0.02},
    ])
    series = _long_stack_monthly_dd(trades)
    assert abs(series.loc["2026-01"] - (-0.02)) < 1e-9
    assert abs(series.loc["2026-02"] - 0.02) < 1e-9
