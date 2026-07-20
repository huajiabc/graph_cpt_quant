import pandas as pd

from pressure_graph.reports.v112_high_vol_topology_continuation import (
    V112Config,
    apply_v112_volatility_gate,
    build_v112_volatility_state,
)


def test_v112_volatility_gate_is_as_of_and_inclusive() -> None:
    times = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"], utc=True)
    portfolios = pd.DataFrame(
        {"feature_time": times, "spread_net_20bp": [0.01, 0.02]}
    )
    state = pd.DataFrame(
        {
            "feature_time": times,
            "month_start": pd.Timestamp("2026-01-01", tz="UTC"),
            "btc_volatility_24h": [0.03, 0.02],
            "btc_volatility_threshold": [0.03, 0.03],
        }
    )
    selected = apply_v112_volatility_gate(portfolios, state)
    assert selected["feature_time"].tolist() == [times[0]]


def test_v112_state_uses_only_pre_month_threshold_history() -> None:
    times = pd.date_range("2025-12-01", "2026-01-03", freq="h", tz="UTC")
    panel = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "feature_time": times,
            "ret_1h": [0.001] * (len(times) - 24) + [0.10] * 24,
        }
    )
    state = build_v112_volatility_state(
        panel,
        [pd.Timestamp("2026-01-01", tz="UTC")],
        V112Config(volatility_hours=2),
    )
    assert not state.empty
    assert state["btc_volatility_threshold"].nunique() == 1
