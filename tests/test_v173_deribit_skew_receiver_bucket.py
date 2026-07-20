from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v173_deribit_skew_receiver_bucket import (
    ALTS,
    BTC,
    V173Config,
    build_monthly_receiver_graph,
    build_surface_signals,
)


def test_surface_signal_uses_only_shifted_robust_history() -> None:
    dates = pd.date_range("2023-01-01", periods=48, freq="D", tz="UTC")
    rr = np.cumsum(np.r_[0.0, np.tile([0.01, -0.01], 23), 0.20])
    surface = pd.DataFrame(
        {
            "quality_pass": True,
            "feature_time": dates,
            "expiration_time": pd.Timestamp("2023-03-31 08:00", tz="UTC"),
            "downside_risk_reversal": rr,
            "atm_iv": np.r_[np.repeat(0.60, 47), 0.70],
        }
    )
    cfg = V173Config(
        robust_lookback=40,
        robust_min_periods=40,
        cooldown_days=1,
    )
    signals = build_surface_signals(surface, cfg)
    assert len(signals) == 1
    assert signals.iloc[0]["event_type"] == "stress"
    assert signals.iloc[0]["feature_time"] == dates[-1]


def test_receiver_graph_prefers_forward_btc_shock_edges() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2022-01-01", periods=2_200, freq="h", tz="UTC")
    btc = rng.normal(0, 0.01, len(index))
    data = {BTC: btc}
    for position, alt in enumerate(ALTS):
        noise = rng.normal(0, 0.003 + position * 0.0001, len(index))
        if position < 4:
            data[alt] = np.sign(rng.normal(size=len(index))) * (
                0.8 * np.r_[0.0, np.abs(btc[:-1])] + np.abs(noise)
            )
        else:
            data[alt] = noise
    returns = pd.DataFrame(data, index=index)
    month = pd.Timestamp("2022-04-01", tz="UTC")
    graph = build_monthly_receiver_graph(
        returns,
        month,
        month,
        V173Config(graph_min_samples=1_000),
    )
    selected = set(graph.loc[graph["selected"], "receiver"])
    assert selected == set(ALTS[:4])
    assert graph.loc[graph["selected"], "sample_n"].ge(1_000).all()
