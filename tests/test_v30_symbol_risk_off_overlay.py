from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v30_symbol_risk_off_overlay import (
    V30Config,
    _overlay_summaries,
    _risk_event_details,
)
from pressure_graph.reports.v12s2_long_risk_off_overlay import RiskOffConfig


def _sample() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    rows = [
        {
            "signal_id": "a",
            "trade_key": "a",
            "exchange": "bybit",
            "symbol": "AAAUSDT",
            "candidate": "CIC1_beta_extreme",
            "signal_time": base + pd.Timedelta(minutes=15),
            "entry_time": base + pd.Timedelta(minutes=30),
            "exit_time": base + pd.Timedelta(hours=3),
            "net_return_at_cost": -0.02,
            "checkpoint_time": base + pd.Timedelta(hours=1, minutes=30),
            "checkpoint_price_covered": True,
            "checkpoint_net_at_cost": -0.01,
            "burst_count_so_far": 10,
            "burst_id": "burst_1",
            "beta_extreme_strength_high": True,
            "cluster_density_high": False,
            "market_impulse_density_high": True,
            "local_volume_shock_strength_high": True,
        },
        {
            "signal_id": "b",
            "trade_key": "b",
            "exchange": "bybit",
            "symbol": "BBBUSDT",
            "candidate": "CIC2_beta_broad",
            "signal_time": base + pd.Timedelta(minutes=30),
            "entry_time": base + pd.Timedelta(minutes=45),
            "exit_time": base + pd.Timedelta(hours=2),
            "net_return_at_cost": 0.03,
            "checkpoint_time": base + pd.Timedelta(hours=1, minutes=45),
            "checkpoint_price_covered": True,
            "checkpoint_net_at_cost": 0.01,
            "burst_count_so_far": 2,
            "burst_id": "burst_1",
            "beta_extreme_strength_high": False,
            "cluster_density_high": False,
            "market_impulse_density_high": True,
            "local_volume_shock_strength_high": False,
        },
    ]
    return pd.DataFrame(rows)


def test_risk_event_details_are_strict_asof() -> None:
    sample = _sample()
    events = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "motif": "S1", "feature_time": sample.loc[0, "signal_time"] - pd.Timedelta(minutes=15)},
            {"symbol": "BBBUSDT", "motif": "S3", "feature_time": sample.loc[1, "signal_time"] + pd.Timedelta(minutes=15)},
        ]
    )
    details = _risk_event_details(sample, events, RiskOffConfig(symbol_cooldown_bars=48))
    assert bool(details.loc[0, "risk_off_active"]) is True
    assert bool(details.loc[1, "risk_off_active"]) is False
    assert details.loc[0, "risk_off_motifs"] == "S1"


def test_overlay_summaries_remove_same_symbol_risk_off_candidate() -> None:
    sample = _sample()
    events = pd.DataFrame(
        [{"symbol": "AAAUSDT", "motif": "S1", "feature_time": sample.loc[0, "signal_time"] - pd.Timedelta(minutes=15)}]
    )
    cfg = V30Config(report_root=Path("unused"))
    summary, skipped = _overlay_summaries(sample, events, cfg)
    row = summary[summary["structure_id"].eq("R0_P2_MAX8")].iloc[0]
    assert int(row["risk_off_gated_candidates"]) == 1
    assert row["risk_off_gated_net20_avg"] < 0
    assert set(skipped["symbol"]) == {"AAAUSDT"}
