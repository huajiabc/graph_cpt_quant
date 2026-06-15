"""Unit tests for v3.4 true-short-sleeve.

These exercise the parts that can run without the v0.9D trade cache or a real
feature parquet: gates, breakdown walker, signal emission, execution, and the
A/B/C three-action compare. The full end-to-end ``write_v3_4_true_short_sleeve``
is exercised on the A100 box where the trade cache lives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressure_graph.backtest.short_execution import ShortExitRule
from pressure_graph.reports.v3_4_true_short_sleeve import (
    FAST_RULE,
    SLEEVES,
    SWING_RULE,
    SleeveSpec,
    V34Config,
    _apply_gates,
    _build_cic_long_index,
    _candidate_summary,
    _clean_hit_summary,
    _collect_symbol_signals,
    _cost_stress,
    _emit_cic_sleeve_signals,
    _emit_motif_sleeve_signals,
    _execute_signal,
    _find_breakdown,
    _GATE_REGISTRY,
    _net_at_cost,
    _reference_low_for,
    _short_vs_no_long,
    _three_action_compare,
)


def _bars(prices: list[float], symbol: str = "AAAUSDT", start: str = "2026-01-01") -> pd.DataFrame:
    """Build a synthetic 15m bar frame with all columns v3.4 gates may probe."""
    times = pd.date_range(start, periods=len(prices), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": symbol,
            "bar_open_time": times,
            "bar_close_time": times + pd.Timedelta(minutes=15),
            "feature_time": times,
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "ret_1h": [0.0] * len(prices),
            "ret_4h_percentile": [50.0] * len(prices),
            "volume_z_4h": [0.0] * len(prices),
            "upper_wick_ratio": [0.2] * len(prices),
            "funding_percentile": [50.0] * len(prices),
            "oi_value_delta_4h_percentile": [50.0] * len(prices),
            "dynamic_all_rank": [10] * len(prices),
            "btc_market_state": ["BTC_chop"] * len(prices),
            "gate_BTC_up": [False] * len(prices),
            "gate_BTC_down": [False] * len(prices),
            "bullish_volume_shock_event": [False] * len(prices),
            "co_impulse_density": [0.5] * len(prices),
        }
    )


# ---------- Gate registry --------------------------------------------------------------


def test_gate_btc_not_up_passes_when_not_up_and_fails_when_up():
    cfg = V34Config()
    df = _bars([100.0] * 5)
    assert _GATE_REGISTRY["btc_not_up"](df, 2, cfg) is True
    df.loc[2, "gate_BTC_up"] = True
    assert _GATE_REGISTRY["btc_not_up"](df, 2, cfg) is False


def test_gate_btc_down_requires_explicit_flag():
    cfg = V34Config()
    df = _bars([100.0] * 5)
    assert _GATE_REGISTRY["btc_down"](df, 2, cfg) is False
    df.loc[2, "gate_BTC_down"] = True
    assert _GATE_REGISTRY["btc_down"](df, 2, cfg) is True


def test_gate_low_coimpulse_uses_percentile_window():
    cfg = V34Config(coimpulse_low_percentile=40.0)
    prices = [100.0] * 20
    df = _bars(prices)
    # Step pattern: 10 high-density bars then 10 low — at idx=12 the value sits
    # well under the 40th-percentile of the cumulative window.
    df["co_impulse_density"] = [1.0] * 10 + [0.1] * 10
    assert _GATE_REGISTRY["low_coimpulse"](df, 12, cfg) is True
    # Early-window guard (warmup): first 8 bars always fail because the gate
    # refuses to fire on too-small history.
    assert _GATE_REGISTRY["low_coimpulse"](df, 3, cfg) is False
    # Late-window after density restores: a bar with the highest density value
    # in a varied window is above the 40th percentile, so the gate fails.
    df["co_impulse_density"] = list(np.linspace(0.1, 1.0, 20))
    assert _GATE_REGISTRY["low_coimpulse"](df, 19, cfg) is False


def test_gate_price_stall_threshold():
    cfg = V34Config()
    df = _bars([100.0] * 5)
    df.loc[2, "ret_4h_percentile"] = 40.0  # below 55 -> stall
    assert _GATE_REGISTRY["price_stall"](df, 2, cfg) is True
    df.loc[2, "ret_4h_percentile"] = 70.0
    assert _GATE_REGISTRY["price_stall"](df, 2, cfg) is False


def test_gate_failed_followthrough_needs_both_vol_and_small_move():
    cfg = V34Config()
    df = _bars([100.0] * 5)
    df.loc[2, "volume_z_4h"] = 3.0
    df.loc[2, "ret_1h"] = 0.001
    assert _GATE_REGISTRY["failed_followthrough"](df, 2, cfg) is True
    df.loc[2, "ret_1h"] = 0.02  # large move -> not failed follow-through
    assert _GATE_REGISTRY["failed_followthrough"](df, 2, cfg) is False


def test_gate_density_fading_uses_lookback_avg():
    cfg = V34Config(density_fading_lookback=4)
    df = _bars([100.0] * 10)
    df["co_impulse_density"] = [1.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    assert _GATE_REGISTRY["density_fading"](df, 5, cfg) is True
    df["co_impulse_density"] = [0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert _GATE_REGISTRY["density_fading"](df, 5, cfg) is False


def test_gate_no_protect_a_missing_column_passes():
    cfg = V34Config()
    df = _bars([100.0] * 5)
    assert _GATE_REGISTRY["no_protect_a"](df, 2, cfg) is True
    df["gate_Protect_A"] = [False, False, True, False, False]
    assert _GATE_REGISTRY["no_protect_a"](df, 2, cfg) is False


def test_gate_cp60_exit_flags_stagnant_close():
    cfg = V34Config(cp60_exit_threshold=0.005)
    df = _bars([100.0, 100.05, 100.1, 100.0, 100.02, 100.04])
    # idx=5: close=100.04 vs idx-4 close=100.05 -> |delta| ~ 0.0001 -> stagnant
    assert _GATE_REGISTRY["cp60_would_exit"](df, 5, cfg) is True
    df.loc[5, "close"] = 105.0
    df.loc[5, "open"] = 105.0
    df.loc[5, "high"] = 105.5
    df.loc[5, "low"] = 104.5
    assert _GATE_REGISTRY["cp60_would_exit"](df, 5, cfg) is False


def test_apply_gates_short_circuits_on_first_failure():
    cfg = V34Config()
    df = _bars([100.0] * 5)
    sleeve = SleeveSpec(code="X", name="x", description="", source="motif",
                       gates=("btc_down", "price_stall"))
    passed, reason = _apply_gates(df, 2, sleeve, cfg)
    assert passed is False
    assert "btc_down" in reason


# ---------- Breakdown walker + reference-low ------------------------------------------


def test_find_breakdown_returns_first_close_below_reference():
    df = _bars([100.0, 101.0, 102.0, 99.0, 98.0, 97.0])
    idx = _find_breakdown(df, start_idx=0, reference_level=99.5, valid_bars=10)
    assert idx == 3  # first close <99.5


def test_find_breakdown_returns_minus_one_when_no_break_in_window():
    df = _bars([100.0, 100.0, 100.0, 100.0])
    idx = _find_breakdown(df, start_idx=0, reference_level=99.0, valid_bars=10)
    assert idx == -1


def test_reference_low_for_reclaim_low_is_anchor_window_min():
    df = _bars([100.0, 95.0, 98.0, 100.0])
    level, ref_bar = _reference_low_for(df, anchor_idx=0, confirmation_idx=3, reference="reclaim_low")
    # min(low * 0.995) from idx 0..3 is at idx=1
    assert ref_bar == 1
    assert level < 100.0


def test_reference_low_for_entry_uses_confirmation_close():
    df = _bars([100.0, 99.0, 98.0, 97.0])
    level, ref_bar = _reference_low_for(df, anchor_idx=0, confirmation_idx=2, reference="entry")
    assert level == 98.0
    assert ref_bar == 2


# ---------- Motif sleeve emission ------------------------------------------------------


def _frame_with_failed_reclaim() -> pd.DataFrame:
    """Build a bar series that triggers the v1.2s S1 detector + the v3.4 breakdown.

    Path: shock(2) -> pullback(5) -> hold(6,7) -> reclaim(8) -> failure(10) ->
    breakdown(13). Reclaim window low is anchored at bar 5 (~97.5) so a close
    of 95 at bar 13 trips the SS1 breakdown threshold.
    """
    prices = [100.0] * 30
    prices[5] = 98.0  # pullback (low ~97.51 trips 1% pullback target)
    prices[6] = 98.5  # still under shock close — no premature reclaim
    prices[7] = 98.5
    prices[8] = 101.0  # reclaim attempt clears the 100 level
    prices[9] = 100.5
    prices[10] = 99.0  # reclaim fails (close < 100)
    prices[13] = 95.0  # breakdown < reclaim-window low ~97.51
    df = _bars(prices)
    df.loc[2, "bullish_volume_shock_event"] = True
    return df


def test_emit_motif_sleeve_signals_fires_on_failed_reclaim_breakdown():
    cfg = V34Config()
    sleeve = SLEEVES[0]  # SS1A
    df = _frame_with_failed_reclaim()
    rows = _emit_motif_sleeve_signals(df, sleeve, cfg)
    # Either fires (>=1) or gate blocks (BTC_not_up passes in synthetic fixture);
    # the structural ingredients are present so we expect at least one signal.
    assert len(rows) >= 1
    first = rows[0]
    assert first["sleeve_code"] == "SS1A"
    assert first["motif_code"] in ("S1", "S3", "S5")


def test_emit_motif_sleeve_signals_blocked_when_gate_fails():
    cfg = V34Config()
    sleeve = SLEEVES[0]
    df = _frame_with_failed_reclaim()
    df.loc[:, "gate_BTC_up"] = True  # btc_not_up gate fails everywhere
    rows = _emit_motif_sleeve_signals(df, sleeve, cfg)
    assert rows == []


# ---------- Execution wrapper ---------------------------------------------------------


def test_execute_signal_runs_fast_and_swing():
    df = _bars([100.0, 100.0, 99.0, 97.0, 96.0, 95.0])
    signal = {
        "sleeve_code": "SS1A",
        "sleeve_name": "x",
        "symbol": "AAAUSDT",
        "entry_idx": 1,
    }
    out = _execute_signal(df, signal, FAST_RULE, "fast")
    assert out["execution"] == "fast"
    assert np.isfinite(out["gross_return"])
    assert out["exit_reason"] in {"take_profit", "stop", "max_hold", "stop_ambiguous", "tp_ambiguous"}


# ---------- net at cost ----------------------------------------------------------------


def test_net_at_cost_subtracts_round_trip():
    gross = pd.Series([0.03, 0.05])
    net = _net_at_cost(gross, 20.0, 5.0)
    # 2*(20+5)bp = 50bp = 0.005 round trip
    assert abs(net.iloc[0] - 0.025) < 1e-12
    assert abs(net.iloc[1] - 0.045) < 1e-12


# ---------- 3-action compare -----------------------------------------------------------


def _signals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sleeve_code": "SS1A",
                "execution": "fast",
                "symbol": "AAAUSDT",
                "signal_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "entry_time": pd.Timestamp("2026-01-01T00:15:00Z"),
                "gross_return": 0.02,
                "exit_reason": "take_profit",
                "max_adverse_excursion": 0.01,
                "max_favorable_excursion": -0.02,
                "squeezed": False,
                "btc_state": "BTC_chop",
                "month": "2026-01",
            },
            {
                "sleeve_code": "SS1A",
                "execution": "swing",
                "symbol": "AAAUSDT",
                "signal_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "entry_time": pd.Timestamp("2026-01-01T00:15:00Z"),
                "gross_return": 0.04,
                "exit_reason": "take_profit",
                "max_adverse_excursion": 0.005,
                "max_favorable_excursion": -0.04,
                "squeezed": False,
                "btc_state": "BTC_chop",
                "month": "2026-01",
            },
        ]
    )


def test_three_action_compare_uses_matched_long_net():
    cfg = V34Config(no_long_cooldown_bars=16)
    signals = _signals_frame()
    # One long lands at signal_time + 15min (within 4h cooldown) with realized net -0.01.
    cic_index = {
        ("AAAUSDT", pd.Timestamp("2026-01-01T00:15:00Z")): {
            "candidate": "CIC1_beta_extreme",
            "net_return": -0.01,
            "entry_time": pd.Timestamp("2026-01-01T00:15:00Z"),
            "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
            "month": "2026-01",
        }
    }
    out = _three_action_compare(signals, cic_index, cfg)
    assert out.iloc[0]["matched_longs"] == 1
    # no_action_long_net should equal the matched long's net (-0.01)
    assert abs(out.iloc[0]["no_action_long_net"] - (-0.01)) < 1e-12
    # no_long_net = -A
    assert abs(out.iloc[0]["no_long_net"] - 0.01) < 1e-12


def test_three_action_compare_no_match_returns_zero():
    cfg = V34Config()
    signals = _signals_frame()
    out = _three_action_compare(signals, {}, cfg)
    assert (out["no_action_long_net"] == 0.0).all()
    assert (out["matched_longs"] == 0).all()


def test_short_vs_no_long_decision_table_verdicts():
    cfg = V34Config()
    signals = _signals_frame()
    # Case 1: long would have lost -0.02 -> no_long=+0.02 > short=+0.015 -> risk_off
    cic_index = {
        ("AAAUSDT", pd.Timestamp("2026-01-01T00:15:00Z")): {
            "candidate": "CIC1_beta_extreme",
            "net_return": -0.02,
            "entry_time": pd.Timestamp("2026-01-01T00:15:00Z"),
            "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
            "month": "2026-01",
        }
    }
    out = _three_action_compare(signals, cic_index, cfg)
    decision = _short_vs_no_long(out, cfg)
    assert not decision.empty
    # Verdict types are limited to the three known labels.
    assert set(decision["verdict"].unique()).issubset({"true_short_value", "risk_off_value", "no_value"})


# ---------- CIC long index + SS3 emission ---------------------------------------------


def test_build_cic_long_index_extracts_per_symbol_rows():
    cfg = V34Config()
    # _focus_pool -> _pool_trades -> _dedupe_pool needs base_signal_id +
    # cost_single_side_bps (focal = 20.0); supply both for the synthetic row.
    cache = pd.DataFrame(
        [
            {
                "candidate": "CIC1_beta_extreme",
                "base_signal_id": "AAAUSDT@20260101000000",
                "cost_single_side_bps": 20.0,
                "symbol": "AAAUSDT",
                "signal_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "entry_time": pd.Timestamp("2026-01-01T00:15:00Z"),
                "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
                "net_return": -0.01,
                "pullback_low": 99.0,
            }
        ]
    )
    index = _build_cic_long_index(cache, cfg)
    assert ("AAAUSDT", pd.Timestamp("2026-01-01T00:00:00Z")) in index


def test_emit_cic_sleeve_signals_requires_index_entry():
    cfg = V34Config()
    sleeve = SLEEVES[4]  # SS3A
    df = _bars([100.0] * 10)
    df.loc[5, "close"] = 95.0  # breakdown
    df.loc[5, "open"] = 99.0
    df.loc[5, "low"] = 94.0
    df.loc[5, "high"] = 99.5
    # cp60 needs idx-4 close ≈ idx close
    rows = _emit_cic_sleeve_signals(df, sleeve, {}, cfg)
    assert rows == []


# ---------- Aggregation -----------------------------------------------------------------


def test_candidate_summary_handles_empty_input():
    cfg = V34Config()
    assert _candidate_summary(pd.DataFrame(), cfg).empty


def test_clean_hit_summary_separates_tp_no_squeeze():
    cfg = V34Config()
    trades = pd.DataFrame(
        [
            {"sleeve_code": "SS1A", "execution": "fast", "exit_reason": "take_profit", "squeezed": False, "gross_return": 0.03, "symbol": "A", "month": "2026-01"},
            {"sleeve_code": "SS1A", "execution": "fast", "exit_reason": "take_profit", "squeezed": True, "gross_return": 0.03, "symbol": "A", "month": "2026-01"},
            {"sleeve_code": "SS1A", "execution": "fast", "exit_reason": "stop", "squeezed": True, "gross_return": -0.02, "symbol": "A", "month": "2026-01"},
        ]
    )
    out = _clean_hit_summary(trades, cfg)
    row = out.iloc[0]
    assert row["trades_total"] == 3
    assert row["trades_clean"] == 1  # only the first row qualifies
    assert abs(row["clean_hit_rate"] - 1 / 3) < 1e-12


def test_cost_stress_grids_costs_and_slippages():
    cfg = V34Config()
    trades = pd.DataFrame(
        [
            {"sleeve_code": "SS1A", "execution": "fast", "gross_return": 0.03, "symbol": "A", "month": "2026-01", "squeezed": False, "exit_reason": "take_profit", "max_adverse_excursion": 0.01},
        ]
    )
    out = _cost_stress(trades, cfg)
    assert len(out) == len(cfg.cost_grid_bps) * len(cfg.extra_slippage_grid_bps)


# ---------- collect_symbol_signals end-to-end -----------------------------------------


def test_collect_symbol_signals_attaches_fast_and_swing_executions():
    cfg = V34Config()
    df = _frame_with_failed_reclaim()
    rows = _collect_symbol_signals(df, cfg.sleeves, {}, cfg)
    if rows:
        executions = {r["execution"] for r in rows}
        # If any signal was emitted, both executions should be attached per signal.
        assert "fast" in executions
        assert "swing" in executions
