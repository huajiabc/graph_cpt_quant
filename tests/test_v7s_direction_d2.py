"""Tests for v7S Direction D2 — CVD-confirmed pair short."""
from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v7s_direction_d2_cvd_pair import (
    CANDIDATE_D2_BTC,
    D2Config,
    DIRECTION_D2,
    _emit_d2_signals,
    _gate_beta_failed_followthrough,
    _gate_beta_overextended,
    _gate_cvd_divergence,
    _gate_relative_overperf,
)


def _group(
    *,
    ret_pcts: list[float],
    closes: list[float],
    highs: list[float] | None = None,
    btc_ret_4hs: list[float] | None = None,
) -> pd.DataFrame:
    n = len(ret_pcts)
    return pd.DataFrame({
        "feature_time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "bar_open_time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "symbol": ["DOGEUSDT"] * n,
        "exchange": ["binance"] * n,
        "btc_market_state": ["BTC_chop"] * n,
        "ret_4h_percentile": ret_pcts,
        "btc_ret_4h": btc_ret_4hs or [0.0] * n,
        "close": closes,
        "open": closes,
        "high": highs or closes,
        "low": [c * 0.99 for c in closes],
    })


class TestD2Gates:
    def test_overextended_fires(self) -> None:
        cfg = D2Config(d2_lookback_bars=4, d2_overextended_pct=95.0)
        g = _group(ret_pcts=[60, 70, 96, 80, 70], closes=[100, 101, 102, 101, 100])
        assert _gate_beta_overextended(g, idx=4, threshold=95.0, cfg=cfg) is True

    def test_relative_overperf_fires_when_beta_beats_btc(self) -> None:
        cfg = D2Config(d2_lookback_bars=4, d2_relative_overperf_min=0.02)
        g = _group(
            ret_pcts=[60] * 5,
            closes=[100, 100, 100, 100, 110],  # beta +10 %
            btc_ret_4hs=[0.0, 0.0, 0.0, 0.0, 0.01],  # BTC +1 %
        )
        assert _gate_relative_overperf(g, idx=4, hedge_lookup=None, cfg=cfg) is True

    def test_relative_overperf_fails_when_beta_lags_btc(self) -> None:
        cfg = D2Config(d2_lookback_bars=4, d2_relative_overperf_min=0.02)
        g = _group(
            ret_pcts=[60] * 5,
            closes=[100, 100, 100, 100, 101],  # beta +1 %
            btc_ret_4hs=[0.0, 0.0, 0.0, 0.0, 0.03],  # BTC +3 %
        )
        assert _gate_relative_overperf(g, idx=4, hedge_lookup=None, cfg=cfg) is False

    def test_failed_follow_through_fires_on_drop(self) -> None:
        cfg = D2Config(d2_lookback_bars=4, d2_reclaim_tolerance=0.015)
        g = _group(
            ret_pcts=[60] * 5,
            closes=[100, 105, 110, 105, 104],
            highs=[100, 105, 110, 105, 104],
        )
        # 104 is ~5.5 % below 110 — passes 1.5 % drop bar.
        assert _gate_beta_failed_followthrough(g, idx=4, cfg=cfg) is True


class TestCvdDivergenceGate:
    def test_fails_closed_when_beta_cvd_missing(self) -> None:
        cfg = D2Config()
        ts = pd.Timestamp("2026-01-01", tz="UTC")
        passed, reason = _gate_cvd_divergence(
            ts,
            beta_cvd_lookup=lambda _: None,
            hedge_cvd_lookup=lambda _: {"buy_sell_imbalance": 0.1},
            cfg=cfg,
        )
        assert passed is False
        assert reason == "beta_cvd_missing"

    def test_fails_closed_when_hedge_cvd_missing(self) -> None:
        cfg = D2Config()
        ts = pd.Timestamp("2026-01-01", tz="UTC")
        passed, reason = _gate_cvd_divergence(
            ts,
            beta_cvd_lookup=lambda _: {"buy_sell_imbalance": -0.20},
            hedge_cvd_lookup=lambda _: None,
            cfg=cfg,
        )
        assert passed is False
        assert reason == "hedge_cvd_missing"

    def test_passes_when_beta_weak_and_hedge_strong(self) -> None:
        cfg = D2Config(d2_beta_cvd_max=-0.05, d2_hedge_cvd_min=-0.05)
        ts = pd.Timestamp("2026-01-01", tz="UTC")
        passed, reason = _gate_cvd_divergence(
            ts,
            beta_cvd_lookup=lambda _: {"buy_sell_imbalance": -0.15},  # beta weakening
            hedge_cvd_lookup=lambda _: {"buy_sell_imbalance": 0.10},  # hedge stable+
            cfg=cfg,
        )
        assert passed is True
        assert reason == "ok"

    def test_fails_when_beta_not_weak(self) -> None:
        cfg = D2Config(d2_beta_cvd_max=-0.05)
        ts = pd.Timestamp("2026-01-01", tz="UTC")
        passed, reason = _gate_cvd_divergence(
            ts,
            beta_cvd_lookup=lambda _: {"buy_sell_imbalance": 0.10},  # beta buying
            hedge_cvd_lookup=lambda _: {"buy_sell_imbalance": 0.10},
            cfg=cfg,
        )
        assert passed is False
        assert reason == "beta_cvd_not_weakening"


class TestEmitSignals:
    def test_emits_when_all_gates_pass(self) -> None:
        cfg = D2Config(
            d2_lookback_bars=4,
            d2_overextended_pct=95.0,
            d2_relative_overperf_min=0.02,
            d2_reclaim_tolerance=0.015,
            d2_beta_cvd_max=-0.05,
            d2_hedge_cvd_min=-0.05,
            d2_cooldown_bars=2,
        )
        g = _group(
            ret_pcts=[60, 70, 96, 80, 70, 60, 60],
            closes=[100, 105, 110, 106, 104, 103, 102],
            highs=[100, 105, 110, 106, 104, 103, 102],
            btc_ret_4hs=[0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01],  # BTC +1%, beta dropped -5%+
        )
        signals = _emit_d2_signals(
            g,
            CANDIDATE_D2_BTC,
            beta_cvd_lookup=lambda _: {"buy_sell_imbalance": -0.15},
            hedge_cvd_lookup=lambda _: {"buy_sell_imbalance": 0.10},
            cfg=cfg,
        )
        # At least one bar in the 7-row group should satisfy: overextended in lookback,
        # beta > btc by 2 % over lookback, failed follow-through, CVD divergence.
        assert len(signals) >= 1
        assert signals[0]["direction"] == DIRECTION_D2
        assert signals[0]["candidate_code"] == CANDIDATE_D2_BTC

    def test_skips_excluded_hedge_symbols(self) -> None:
        cfg = D2Config(d2_exclude_symbols=("BTCUSDT",))
        g = _group(ret_pcts=[96] * 5, closes=[100] * 5)
        g["symbol"] = "BTCUSDT"
        signals = _emit_d2_signals(
            g,
            CANDIDATE_D2_BTC,
            beta_cvd_lookup=lambda _: {"buy_sell_imbalance": -0.20},
            hedge_cvd_lookup=lambda _: {"buy_sell_imbalance": 0.10},
            cfg=cfg,
        )
        assert signals == []
