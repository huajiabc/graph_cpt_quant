from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v4s_failure_state_graph import (
    ACTION_ALLOW_LONG,
    ACTION_DISABLE_OVERFLOW,
    ACTION_DISABLE_PROTECT,
    ACTION_EXIT_EXISTING_LONG,
    ACTION_NO_LONG,
    ACTION_NORMAL_SHORT,
    ACTION_SMALL_SHORT,
    ACTIONS,
    PATH_A,
    PATH_B,
    PATH_C,
    PATH_A_SLEEVES,
    PATH_B_SLEEVES,
    V4SConfig,
    _build_atlas,
    _emit_path_c_states,
    _evaluate_actions_for_state,
    _gate_crowded_long_combo,
    _match_active_long,
    _matched_nearby_long,
)


def _synth_group(n: int = 30, *, funding=80.0, oi=75.0, ret_pct=30.0) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAAUSDT"] * n,
        "exchange": ["bybit"] * n,
        "bar_open_time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "feature_time": pd.date_range("2026-01-01 00:15", periods=n, freq="15min", tz="UTC"),
        "close": np.linspace(100, 105, n),
        "open": np.linspace(100, 105, n),
        "high": np.linspace(101, 106, n),
        "low": np.linspace(99, 104, n),
        "funding_percentile": np.full(n, funding),
        "oi_value_delta_4h_percentile": np.full(n, oi),
        "ret_4h_percentile": np.full(n, ret_pct),
        "volume_z_4h": np.full(n, 2.5),
        "ret_1h": np.full(n, 0.001),
        "gate_BTC_up": np.zeros(n, dtype=bool),
        "gate_BTC_down": np.ones(n, dtype=bool),
        "co_impulse_density": np.linspace(0.1, 0.9, n),
    })


def test_actions_list_is_seven() -> None:
    assert ACTIONS == (
        ACTION_ALLOW_LONG,
        ACTION_NO_LONG,
        ACTION_DISABLE_OVERFLOW,
        ACTION_DISABLE_PROTECT,
        ACTION_EXIT_EXISTING_LONG,
        ACTION_SMALL_SHORT,
        ACTION_NORMAL_SHORT,
    )


def test_path_a_uses_two_cic_breakdown_sleeves() -> None:
    assert tuple(s.source for s in PATH_A_SLEEVES) == ("cic_candidate", "cic_candidate")
    assert tuple(s.breakdown_reference for s in PATH_A_SLEEVES) == ("entry", "pullback_low")


def test_path_b_uses_motif_with_reclaim_low_breakdown() -> None:
    sleeve = PATH_B_SLEEVES[0]
    assert sleeve.source == "motif"
    assert sleeve.breakdown_reference == "reclaim_low"
    assert set(sleeve.source_motifs) == {"S1", "S3", "S5"}


def test_path_c_combo_fires_when_all_predicates_hold() -> None:
    df = _synth_group()
    cfg = V4SConfig()
    fires = [_gate_crowded_long_combo(df, i, cfg) for i in range(len(df))]
    assert sum(fires) == len(df)  # synthetic frame satisfies every bar


def test_path_c_combo_drops_when_funding_below_threshold() -> None:
    df = _synth_group(funding=50.0)
    cfg = V4SConfig()
    fires = [_gate_crowded_long_combo(df, i, cfg) for i in range(len(df))]
    assert sum(fires) == 0


def test_path_c_cooldown_caps_emissions() -> None:
    df = _synth_group(n=40)
    cfg = V4SConfig(path_c_cooldown_bars=16)
    states = _emit_path_c_states(df, cfg)
    # n=40 bars, cooldown=16; first valid idx is 1, then 1+16=17, then 17+16=33
    assert len(states) == 3
    for state in states:
        assert state["sleeve_code"] == "C1_crowded_long_stall"


def test_matched_nearby_long_finds_recent_long() -> None:
    by_sym = {
        "AAAUSDT": [
            (int(pd.Timestamp("2026-01-02 09:00", tz="UTC").value), {"net_return": -0.01, "entry_time": pd.Timestamp("2026-01-02 09:00", tz="UTC"), "exit_time": pd.Timestamp("2026-01-02 13:00", tz="UTC")}),
        ]
    }
    found = _matched_nearby_long("AAAUSDT", pd.Timestamp("2026-01-02 10:00", tz="UTC"), by_sym, lookback_ns=int(12 * 3600 * 1e9))
    assert found is not None
    assert found["net_return"] == -0.01


def test_match_active_long_requires_signal_inside_window() -> None:
    by_sym = {
        "AAAUSDT": [
            (int(pd.Timestamp("2026-01-02 09:00", tz="UTC").value), {"entry_time": pd.Timestamp("2026-01-02 09:00", tz="UTC"), "exit_time": pd.Timestamp("2026-01-02 13:00", tz="UTC"), "net_return": -0.01}),
        ]
    }
    inside = _match_active_long("AAAUSDT", pd.Timestamp("2026-01-02 11:00", tz="UTC"), {}, by_sym)
    after_exit = _match_active_long("AAAUSDT", pd.Timestamp("2026-01-02 14:00", tz="UTC"), {}, by_sym)
    assert inside is not None
    assert after_exit is None


def test_evaluate_actions_for_state_with_matched_long_and_short_pnl() -> None:
    state = {
        "symbol": "AAAUSDT",
        "signal_time": pd.Timestamp("2026-01-02 10:00", tz="UTC"),
        "gross_return": 0.025,
    }
    by_sym = {
        "AAAUSDT": [
            (
                int(pd.Timestamp("2026-01-02 09:00", tz="UTC").value),
                {
                    "entry_time": pd.Timestamp("2026-01-02 09:00", tz="UTC"),
                    "exit_time": pd.Timestamp("2026-01-02 13:00", tz="UTC"),
                    "net_return": -0.012,
                    "entry_price": 100.0,
                    "symbol": "AAAUSDT",
                },
            )
        ]
    }
    mark_table = {
        "AAAUSDT": pd.DataFrame(
            {
                "mark_time": pd.date_range("2026-01-02 08:00", periods=20, freq="15min", tz="UTC"),
                "mark_price": [100.0 + i * 0.1 for i in range(20)],
            }
        )
    }
    cfg = V4SConfig()
    out = _evaluate_actions_for_state(state, by_sym, mark_table, long_lookback_ns=12 * 3600 * 10**9, cfg=cfg)
    assert out[ACTION_NO_LONG] == 0.0
    assert out[ACTION_ALLOW_LONG] == -0.012
    assert out[ACTION_DISABLE_OVERFLOW] == -0.012
    # Short with size 0.5 vs 1.0 at 20bp focal cost: gross 0.025, net = 0.025 - 0.004 = 0.021
    assert abs(out[ACTION_NORMAL_SHORT] - 0.021) < 1e-9
    assert abs(out[ACTION_SMALL_SHORT] - 0.0105) < 1e-9
    # exit_existing_long picks up unrealized PnL > matched realized when held to natural exit
    assert out["matched_long"] is True
    assert out["active_long_at_observation"] is True


def test_evaluate_actions_for_state_without_long_falls_back_to_short_only() -> None:
    state = {
        "symbol": "AAAUSDT",
        "signal_time": pd.Timestamp("2026-01-02 10:00", tz="UTC"),
        "gross_return": 0.015,
    }
    cfg = V4SConfig()
    out = _evaluate_actions_for_state(state, {}, {}, long_lookback_ns=10**12, cfg=cfg)
    assert np.isnan(out[ACTION_ALLOW_LONG])
    assert out[ACTION_NO_LONG] == 0.0
    assert abs(out[ACTION_NORMAL_SHORT] - (0.015 - 0.004)) < 1e-9


def test_build_atlas_expands_each_state_into_seven_action_rows() -> None:
    states = pd.DataFrame(
        [
            {
                "path": PATH_C,
                "sleeve_code": "C1_crowded_long_stall",
                "symbol": "AAAUSDT",
                "signal_time": pd.Timestamp("2026-01-02 10:00", tz="UTC"),
                "entry_time": pd.Timestamp("2026-01-02 10:15", tz="UTC"),
                "execution": "fast",
                "gross_return": 0.02,
            }
        ]
    )
    atlas = _build_atlas(states, {}, {}, V4SConfig())
    assert len(atlas) == len(ACTIONS)
    assert set(atlas["action"]) == set(ACTIONS)
