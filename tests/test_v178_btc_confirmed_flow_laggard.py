from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    BTC,
    RAW_CANDIDATE,
    V178Config,
    build_btc_source_signals,
    build_monthly_btc_receiver_graph,
    build_v178_events,
)


def test_source_thresholds_are_shifted_before_decision() -> None:
    index = pd.date_range("2025-01-01", periods=30, freq="15min", tz="UTC")
    close = pd.Series(100.0, index=index)
    close.iloc[-1] = 103.0
    source = pd.DataFrame(
        {
            "close": close,
            "turnover": [100.0] * 29 + [1_000.0],
            "taker_buy_quote": [50.0] * 29 + [900.0],
        },
        index=index,
    )
    cfg = V178Config(
        source_lookback_bars=20,
        source_min_bars=20,
        cooldown_bars=1,
    )
    signals = build_btc_source_signals(source, cfg)
    assert len(signals) == 1
    assert signals.iloc[0]["feature_time"] == index[-1]
    assert signals.iloc[0]["direction"] == 1


def test_graph_finds_engineered_btc_to_alt_lag() -> None:
    rng = np.random.default_rng(3)
    index = pd.date_range("2025-01-01", periods=3_000, freq="15min", tz="UTC")
    btc = rng.normal(0, 0.003, len(index))
    data = {BTC: btc}
    for position in range(12):
        name = f"A{position}USDT"
        noise = rng.normal(0, 0.001, len(index))
        data[name] = np.r_[0.0, btc[:-1] * (0.8 if position < 10 else 0.0)] + noise
    returns = pd.DataFrame(data, index=index)
    month = pd.Timestamp("2025-02-01", tz="UTC")
    graph = build_monthly_btc_receiver_graph(
        returns,
        month,
        month,
        V178Config(graph_min_samples=2_000),
    )
    assert graph["graph_month"].dt.minute.eq(0).all()
    selected = set(graph.loc[graph["selected"], "receiver"])
    assert selected == {f"A{position}USDT" for position in range(10)}


def test_event_selects_nonpositive_directed_residual_laggards() -> None:
    entry = pd.Timestamp("2026-01-02", tz="UTC")
    exit_time = entry + pd.Timedelta(minutes=30)
    symbols = [BTC, "A", "B", "C", "D"]
    close = pd.DataFrame(
        [[100.0] * 5, [101.0, 102.0, 102.0, 102.0, 102.0]],
        index=[entry, exit_time],
        columns=symbols,
    )
    returns = pd.DataFrame(
        [[0.01, 0.0, 0.002, 0.004, 0.02]],
        index=[entry],
        columns=symbols,
    )
    graph = pd.DataFrame(
        {
            "graph_month": pd.Timestamp("2026-01-01", tz="UTC"),
            "receiver": ["A", "B", "C", "D"],
            "btc_beta": [1.0, 1.0, 1.0, 1.0],
            "selected": [True] * 4,
            "receiver_rank": [1, 2, 3, 4],
        }
    )
    signals = pd.DataFrame(
        {"feature_time": [entry], "direction": [1.0], "return_quantile": [0.975]}
    )
    events, selected = build_v178_events(
        signals,
        graph,
        close,
        returns,
        V178Config(min_laggards=3, max_laggards=3),
    )
    assert set(selected["receiver"]) == {"A", "B", "C"}
    raw = events[events["candidate"].eq(RAW_CANDIDATE)].iloc[0]
    assert raw["gross_return"] == pytest.approx(0.02)
