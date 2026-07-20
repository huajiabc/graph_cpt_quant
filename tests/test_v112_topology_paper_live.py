from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.paper_live.v112 import (
    V112LiveConfig,
    build_v112_live_communities,
    build_v112_live_month,
    build_v112_portfolio_ledger,
    build_v112_return_panel,
    merge_v112_signal_ledger,
)


def _cfg() -> V112LiveConfig:
    return V112LiveConfig(
        live_root=Path("data/live"),
        history_days=45,
        base_config=Path("configs/v0_3.yaml"),
        seed_base_events=Path("seed.parquet"),
        report_root=Path("reports/live"),
        symbols=("BTCUSDT", "A", "B", "C", "D", "E", "F"),
        fallback_candidates=("G",),
        community_count=2,
        expected_community_size=3,
    )


def test_v112_return_panel_matches_four_bar_return() -> None:
    times = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    klines = pd.DataFrame(
        {
            "symbol": "A",
            "bar_open_time": times,
            "bar_close_time": times + pd.Timedelta(minutes=15),
            "close": [100, 101, 102, 103, 110, 121],
        }
    )
    panel = build_v112_return_panel(klines)
    assert np.isclose(panel.iloc[4]["ret_1h"], 0.10)
    assert np.isclose(panel.iloc[5]["ret_1h"], 0.19801980198019803)
    assert "turnover_1h" in panel.columns


def test_v112_live_communities_match_balanced_planted_groups() -> None:
    rng = np.random.default_rng(4)
    left = rng.normal(size=700)
    right = rng.normal(size=700)
    residual = pd.DataFrame(
        {
            **{f"L{i}": left + rng.normal(0, 0.05, 700) for i in range(6)},
            **{f"R{i}": right + rng.normal(0, 0.05, 700) for i in range(6)},
        }
    )
    communities = build_v112_live_communities(residual, 2, 500)
    assert {frozenset(group) for group in communities} == {
        frozenset(f"L{i}" for i in range(6)),
        frozenset(f"R{i}" for i in range(6)),
    }


def test_v112_live_month_uses_frozen_fallback_for_incomplete_member() -> None:
    rng = np.random.default_rng(7)
    history_times = pd.date_range(
        "2025-12-02 20:00", periods=700, freq="h", tz="UTC"
    )
    target_times = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    frames = []
    for symbol in ("BTCUSDT", "A", "B", "C", "D", "E", "G"):
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "feature_time": history_times.append(target_times),
                    "ret_1h": rng.normal(0, 0.01, 724),
                    "turnover_1h": 2.0 if symbol == "G" else 1.0,
                }
            )
        )
    frames.append(
        pd.DataFrame(
            {
                "symbol": "F",
                "feature_time": history_times[-100:].append(target_times),
                "ret_1h": rng.normal(0, 0.01, 124),
                "turnover_1h": 1.0,
            }
        )
    )
    _, membership, metadata = build_v112_live_month(pd.concat(frames), _cfg())
    assert metadata["exact_universe_ready"] is False
    assert metadata["replacements"] == ("F->G",)
    assert metadata["community_sizes"] == (3, 3)
    assert set(membership["symbol"]) == {"A", "B", "C", "D", "E", "G"}


def test_v112_signal_ledger_preserves_first_observation() -> None:
    selected = pd.DataFrame(
        {
            "event_id": ["v112-base|one"],
            "feature_time": [pd.Timestamp("2026-01-01 00:00", tz="UTC")],
            "portfolio_id": ["p1"],
        }
    )
    first = merge_v112_signal_ledger(
        pd.DataFrame(), selected, pd.Timestamp("2026-01-01 00:30", tz="UTC"), 60
    )
    second = merge_v112_signal_ledger(
        first, selected, pd.Timestamp("2026-01-01 02:00", tz="UTC"), 60
    )
    assert second.loc[0, "first_observed_at_utc"] == pd.Timestamp(
        "2026-01-01 00:30", tz="UTC"
    )
    assert bool(second.loc[0, "timely_forward_observation"])


def test_v112_portfolio_return_uses_future_four_hours_and_cost() -> None:
    times = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    returns = pd.DataFrame(
        {
            "BTCUSDT": 0.0,
            "A": [0.0, 0.01, 0.01, 0.01, 0.01],
            "B": [0.0, 0.01, 0.01, 0.01, 0.01],
            "C": [0.0, -0.01, -0.01, -0.01, -0.01],
            "D": [0.0, -0.01, -0.01, -0.01, -0.01],
        },
        index=times,
    )
    signals = pd.DataFrame(
        {
            "portfolio_id": ["p1"],
            "signal_id": ["s1"],
            "feature_time": [times[0]],
            "top_symbols": ["A|B"],
            "bottom_symbols": ["C|D"],
            "top_beta": [1.0],
            "bottom_beta": [1.0],
            "timely_forward_observation": [True],
        }
    )
    ledger = build_v112_portfolio_ledger(signals, returns, _cfg())
    expected = 0.5 * ((1.01**4 - 1) - (0.99**4 - 1))
    assert np.isclose(ledger.loc[0, "gross_return_4h"], expected)
    assert np.isclose(ledger.loc[0, "net_return_20bp"], expected - 0.002)
