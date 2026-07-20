from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    V180Config,
    build_v180_events,
    build_v180_signals,
)


def test_dispersion_threshold_uses_only_prior_bars() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    names = [f"A{i}USDT" for i in range(30)]
    returns = pd.DataFrame(0.0, index=index, columns=["BTCUSDT", *names])
    base = np.linspace(-0.001, 0.001, len(names))
    returns.loc[:, names] = np.tile(base, (len(index), 1))
    returns.loc[index[-1], names] = np.linspace(-0.03, 0.03, len(names))
    graph = pd.DataFrame(
        {
            "graph_month": pd.Timestamp("2026-01-01", tz="UTC"),
            "receiver": names,
            "btc_beta": 0.0,
        }
    )
    cfg = V180Config(
        dispersion_lookback_bars=20,
        dispersion_min_bars=20,
        min_cross_section=30,
        cooldown_bars=1,
    )
    signals = build_v180_signals(returns, graph, cfg)
    signal = signals.iloc[-1]
    assert signal["feature_time"] == index[-1]
    assert signal["dispersion"] > signal["dispersion_threshold"]


def test_event_formula_is_beta_hedged_compression() -> None:
    entry = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    exit_time = entry + pd.Timedelta(minutes=15)
    laggards = [f"L{i}USDT" for i in range(5)]
    leaders = [f"H{i}USDT" for i in range(5)]
    close = pd.DataFrame(
        100.0,
        index=pd.DatetimeIndex([entry, exit_time]),
        columns=["BTCUSDT", *laggards, *leaders],
    )
    close.loc[exit_time, "BTCUSDT"] = 101.0
    close.loc[exit_time, laggards] = 102.0
    close.loc[exit_time, leaders] = 100.5
    signals = pd.DataFrame(
        {
            "feature_time": [entry],
            "source_feature_time": [entry],
            "graph_month": [pd.Timestamp("2026-01-01", tz="UTC")],
            "dispersion": [0.05],
            "dispersion_threshold": [0.04],
            "dispersion_quantile": [0.975],
            "cross_section": [30],
            "laggards": ["|".join(laggards)],
            "leaders": ["|".join(leaders)],
            "mean_laggard_beta": [0.5],
            "mean_leader_beta": [0.2],
            "spread_beta": [0.15],
            "mean_laggard_current_residual": [-0.02],
            "mean_leader_current_residual": [0.02],
        }
    )
    events = build_v180_events(signals, close)
    expected = (0.5 * (0.02 - 0.005) - 0.15 * 0.01) / 1.15
    assert math.isclose(events.loc[0, "gross_return"], expected)


def test_delay_keeps_original_membership_and_moves_entry() -> None:
    source = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    index = pd.date_range(source, periods=3, freq="15min", tz="UTC")
    laggards = [f"L{i}USDT" for i in range(5)]
    leaders = [f"H{i}USDT" for i in range(5)]
    close = pd.DataFrame(
        100.0,
        index=index,
        columns=["BTCUSDT", *laggards, *leaders],
    )
    signals = pd.DataFrame(
        {
            "feature_time": [source],
            "source_feature_time": [source],
            "graph_month": [pd.Timestamp("2026-01-01", tz="UTC")],
            "dispersion": [0.05],
            "dispersion_threshold": [0.04],
            "dispersion_quantile": [0.975],
            "cross_section": [30],
            "laggards": ["|".join(laggards)],
            "leaders": ["|".join(leaders)],
            "mean_laggard_beta": [0.5],
            "mean_leader_beta": [0.2],
            "spread_beta": [0.15],
            "mean_laggard_current_residual": [-0.02],
            "mean_leader_current_residual": [0.02],
        }
    )
    delayed = build_v180_events(signals, close, entry_delay_bars=1)
    assert delayed.loc[0, "entry_time"] == index[1]
    assert delayed.loc[0, "laggards"] == "|".join(laggards)
    assert delayed.loc[0, "leaders"] == "|".join(leaders)
