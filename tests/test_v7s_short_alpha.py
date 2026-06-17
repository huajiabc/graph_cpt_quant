"""Unit tests for v7S Short Alpha Exploration.

These tests do not require the v0.9D capacity trade cache or live feature
parquet — they exercise the new gates, the verdict evaluator, and the
no-data driver path with synthetic inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.v7s_short_alpha import (
    CANDIDATE_E1,
    DIRECTION_E,
    GATE_NAMES,
    V7SConfig,
    _evaluate_gates,
    _gate_beta_high_gone,
    _gate_sell_flow_confirms,
    _matched_random_baseline,
    _month_cap_leave_one_month,
    _short_candidate_summary,
    _symbol_contribution,
    write_v7s_short_alpha,
)


def _make_group(beta_history: list[bool], current_beta: bool) -> pd.DataFrame:
    n = len(beta_history) + 1
    return pd.DataFrame(
        {
            "feature_time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
            "bar_open_time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
            "symbol": ["BTCUSDT"] * n,
            "gate_beta_already_extended": beta_history + [current_beta],
        }
    )


class TestBetaHighGoneGate:
    def test_fires_when_beta_was_high_and_is_now_low(self) -> None:
        cfg = V7SConfig(e_beta_high_lookback_bars=4)
        group = _make_group([True, True, False, False], current_beta=False)
        assert _gate_beta_high_gone(group, idx=4, cfg=cfg) is True

    def test_does_not_fire_when_still_high(self) -> None:
        cfg = V7SConfig(e_beta_high_lookback_bars=4)
        group = _make_group([True, True, True, True], current_beta=True)
        assert _gate_beta_high_gone(group, idx=4, cfg=cfg) is False

    def test_does_not_fire_when_never_high(self) -> None:
        cfg = V7SConfig(e_beta_high_lookback_bars=4)
        group = _make_group([False, False, False, False], current_beta=False)
        assert _gate_beta_high_gone(group, idx=4, cfg=cfg) is False

    def test_fails_closed_on_missing_columns(self) -> None:
        cfg = V7SConfig()
        group = pd.DataFrame({"symbol": ["BTCUSDT"]})
        assert _gate_beta_high_gone(group, idx=0, cfg=cfg) is False


class TestSellFlowGate:
    def test_fail_closed_when_no_lookup(self) -> None:
        cfg = V7SConfig(e_sell_flow_fail_open=False)
        group = pd.DataFrame({"symbol": ["BTCUSDT"], "feature_time": pd.date_range("2026-01-01", periods=1, tz="UTC")})
        passed, reason = _gate_sell_flow_confirms(group, idx=0, cfg=cfg, orderflow_lookup=None)
        assert passed is False
        assert reason == "orderflow_missing"

    def test_fail_open_when_configured(self) -> None:
        cfg = V7SConfig(e_sell_flow_fail_open=True)
        group = pd.DataFrame({"symbol": ["BTCUSDT"], "feature_time": pd.date_range("2026-01-01", periods=1, tz="UTC")})
        passed, reason = _gate_sell_flow_confirms(group, idx=0, cfg=cfg, orderflow_lookup=None)
        assert passed is True
        assert reason == "orderflow_missing_open"

    def test_passes_when_imbalance_below_threshold(self) -> None:
        cfg = V7SConfig(e_sell_flow_max_imbalance=-0.05, e_sell_flow_window="reclaim_bar")
        group = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "feature_time": pd.date_range("2026-01-01", periods=1, tz="UTC"),
            }
        )

        def lookup(symbol: str, signal_time: pd.Timestamp) -> dict | None:
            assert symbol == "BTCUSDT"
            return {"reclaim_bar": {"buy_sell_imbalance": -0.20}}

        passed, reason = _gate_sell_flow_confirms(group, idx=0, cfg=cfg, orderflow_lookup=lookup)
        assert passed is True
        assert reason == "ok"

    def test_fails_when_imbalance_above_threshold(self) -> None:
        cfg = V7SConfig(e_sell_flow_max_imbalance=-0.05, e_sell_flow_window="reclaim_bar")
        group = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "feature_time": pd.date_range("2026-01-01", periods=1, tz="UTC"),
            }
        )

        def lookup(symbol: str, signal_time: pd.Timestamp) -> dict | None:
            return {"reclaim_bar": {"buy_sell_imbalance": 0.10}}

        passed, reason = _gate_sell_flow_confirms(group, idx=0, cfg=cfg, orderflow_lookup=lookup)
        assert passed is False
        assert reason == "ok"


def _synth_trades(n_winners: int, n_losers: int) -> pd.DataFrame:
    """Build a multi-symbol synthetic trade frame so symbol/month caps don't trip."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ARBUSDT", "OPUSDT", "INJUSDT"]
    rows: list[dict[str, object]] = []
    for i in range(n_winners):
        rows.append(
            {
                "direction": DIRECTION_E,
                "candidate_code": CANDIDATE_E1,
                "execution": "fast",
                "symbol": symbols[i % len(symbols)],
                "month": f"2026-{(i % 6) + 1:02d}",
                "gross_return": 0.06,
                "net20": 0.05,
                "net30": 0.04,
                "holding_bars": 8,
                "signal_time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
                "entry_idx": 10,
            }
        )
    for i in range(n_losers):
        rows.append(
            {
                "direction": DIRECTION_E,
                "candidate_code": CANDIDATE_E1,
                "execution": "fast",
                "symbol": symbols[i % len(symbols)],
                "month": f"2026-{(i % 6) + 1:02d}",
                "gross_return": -0.03,
                "net20": -0.04,
                "net30": -0.05,
                "holding_bars": 8,
                "signal_time": pd.Timestamp("2026-02-01", tz="UTC") + pd.Timedelta(hours=i),
                "entry_idx": 10,
            }
        )
    return pd.DataFrame(rows)


class TestVerdictEvaluator:
    def test_summary_carries_verdict_for_synthetic_winners(self) -> None:
        cfg = V7SConfig(
            random_baseline_draws=20,
            random_baseline_seed=1,
            max_squeeze_share=0.20,
            month_cap_pct=0.35,
            max_symbol_share=0.35,
            hedge_corr_max=-0.30,
        )
        trades = _synth_trades(n_winners=40, n_losers=10)
        summary = _short_candidate_summary(trades, cfg)
        assert not summary.empty
        baseline = _matched_random_baseline(trades, cfg)
        month_cap = _month_cap_leave_one_month(trades, cfg)
        symbol = _symbol_contribution(trades, cfg)

        # Stand-in tables for the other inputs (most gates require their row).
        cost_grid = pd.DataFrame(
            [
                {
                    "direction": DIRECTION_E,
                    "candidate_code": CANDIDATE_E1,
                    "execution": "fast",
                    "cost_bps": 30.0,
                    "extra_slippage_bps": 0.0,
                    "n_trades": 50,
                    "mean_net": 0.025,
                    "win_rate": 0.6,
                }
            ]
        )
        first_touch = pd.DataFrame(
            [
                {
                    "direction": DIRECTION_E,
                    "candidate_code": CANDIDATE_E1,
                    "execution": "fast",
                    "n": 50,
                    "hit_down_3pct": 0.60,
                    "hit_down_5pct": 0.40,
                    "up_before_down_2pct": 0.10,
                    "short_squeeze_before_hit": 0.08,
                    "max_adverse_up_mean": 0.02,
                }
            ]
        )
        vs_no_long = pd.DataFrame(
            [
                {
                    "direction": DIRECTION_E,
                    "candidate_code": CANDIDATE_E1,
                    "execution": "fast",
                    "n": 50,
                    "mean_A_no_action": 0.005,
                    "mean_B_no_long": 0.0,
                    "mean_C_short": 0.02,
                    "short_beats_no_long_pct": 0.7,
                }
            ]
        )
        hedge = pd.DataFrame(
            [
                {
                    "direction": DIRECTION_E,
                    "candidate_code": CANDIDATE_E1,
                    "execution": "fast",
                    "n_months": 6,
                    "hedge_corr": -0.55,
                    "long_worst_month": "2026-03",
                    "short_in_long_worst_month": 0.04,
                }
            ]
        )

        evaluated = _evaluate_gates(
            summary, cost_grid, first_touch, vs_no_long, hedge, month_cap, symbol, baseline, cfg
        )
        assert set(GATE_NAMES).issubset(evaluated.columns)
        verdicts = set(evaluated["verdict"].unique())
        # With strongly positive synthetic returns + hedge negative corr, we expect promote.
        assert "promote" in verdicts


class TestDriverNoData:
    def test_no_data_writes_stub_files(self, tmp_path: Path) -> None:
        cfg = V7SConfig(
            report_root=tmp_path / "v7s_short_alpha",
            trade_cache_path=tmp_path / "missing_trade_cache.parquet",
        )
        # Feature path also does not exist.
        outputs = write_v7s_short_alpha(
            feature_path=tmp_path / "missing_features.parquet",
            instruments=pd.DataFrame(),
            config=None,
            cfg=cfg,
        )
        assert outputs["summary"].exists()
        assert outputs["candidate_notes"].exists()
        notes = outputs["candidate_notes"].read_text(encoding="utf-8")
        assert "No data" in notes
        # All ten standardized CSV outputs must exist as stubs.
        for key in (
            "summary",
            "cost_grid",
            "first_touch",
            "vs_no_long",
            "vs_exit_long",
            "hedge",
            "month_cap",
            "symbol_contrib",
            "baseline",
        ):
            assert outputs[key].exists(), f"missing stub: {key}"


class TestUnimplementedDirections:
    def test_direction_a_raises_not_implemented(self, tmp_path: Path) -> None:
        cfg = V7SConfig(
            report_root=tmp_path / "v7s_short_alpha",
            enabled_directions=("A_cross_exchange_lag",),
        )
        with pytest.raises(NotImplementedError, match="Direction A"):
            write_v7s_short_alpha(
                feature_path=tmp_path / "missing.parquet",
                instruments=pd.DataFrame(),
                config=None,
                cfg=cfg,
            )
