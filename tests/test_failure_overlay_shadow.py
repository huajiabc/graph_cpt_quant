from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.failure_overlay_shadow import (
    STRATEGIES,
    STRATEGY_F3,
    STRATEGY_F5,
    ShadowConfig,
    _build_ledger,
    _daily_summary,
    _f3_decisions,
    _f5_decisions,
    _flag_overflow_candidate,
    _strategy_status,
    _verdict_band,
)
from pressure_graph.reports.v3_5_failure_risk_layer_bridge import (
    ACTION_ALLOW,
    ACTION_SKIP_FULL,
    ACTION_SKIP_OVERFLOW,
)


def _toy_pool(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    if "candidate" not in df.columns:
        df["candidate"] = "CIC1_beta_extreme"
    if "burst_count_so_far" not in df.columns:
        df["burst_count_so_far"] = 0
    if "month" not in df.columns:
        df["month"] = df["entry_time"].dt.strftime("%Y-%m")
    df["candidate"] = df["candidate"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce").fillna(0.0)
    return df


def _events(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": sym, "motif": motif, "feature_time": pd.Timestamp(ts, tz="UTC")}
            for sym, motif, ts in rows
        ]
    )


def test_two_strategies_registered() -> None:
    assert STRATEGIES == (STRATEGY_F3, STRATEGY_F5)


def test_flag_overflow_candidate_respects_burst_threshold() -> None:
    pool = _toy_pool(
        [
            {"symbol": "X", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": 0.01, "burst_count_so_far": 8},
            {"symbol": "Y", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": 0.02, "burst_count_so_far": 9},
            {"symbol": "Z", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": -0.03, "burst_count_so_far": 12},
        ]
    )
    flag = _flag_overflow_candidate(pool, ShadowConfig())
    assert list(flag) == [False, True, True]


def test_f3_blocks_only_overflow_eligible_under_failure_recent() -> None:
    pool = _toy_pool(
        [
            {"symbol": "X", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": 0.01, "burst_count_so_far": 4},
            {"symbol": "X", "signal_time": "2026-01-02 09:30", "entry_time": "2026-01-02 09:45",
             "exit_time": "2026-01-02 13:45", "net_return": 0.02, "burst_count_so_far": 12},
        ]
    )
    failure_recent = pd.Series([True, True], index=pool.index)
    decisions = _f3_decisions(pool, failure_recent, ShadowConfig())
    assert decisions.iloc[0] == ACTION_ALLOW  # not overflow-eligible
    assert decisions.iloc[1] == ACTION_SKIP_OVERFLOW  # overflow + failure_recent


def test_f5_blocks_only_cic2_under_failure_recent() -> None:
    pool = _toy_pool(
        [
            {"symbol": "X", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": 0.01,
             "candidate": "CIC1_beta_extreme", "burst_count_so_far": 12},
            {"symbol": "X", "signal_time": "2026-01-02 09:30", "entry_time": "2026-01-02 09:45",
             "exit_time": "2026-01-02 13:45", "net_return": -0.02,
             "candidate": "CIC2_beta_broad", "burst_count_so_far": 12},
        ]
    )
    failure_recent = pd.Series([True, True], index=pool.index)
    decisions = _f5_decisions(pool, failure_recent, ShadowConfig())
    assert decisions.iloc[0] == ACTION_ALLOW  # CIC1 — not gated
    assert decisions.iloc[1] == ACTION_SKIP_FULL  # CIC2 + failure_recent


def test_build_ledger_emits_two_rows_per_pool_row() -> None:
    pool = _toy_pool(
        [
            {"symbol": "X", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": -0.02,
             "candidate": "CIC2_beta_broad", "burst_count_so_far": 12},
        ]
    )
    events = _events([("X", "S1", "2026-01-02 08:00")])
    cfg = ShadowConfig(cooldown_bars=8)
    ledger = _build_ledger(pool, events, cfg)
    # one row per strategy
    assert set(ledger["strategy"]) == {STRATEGY_F3, STRATEGY_F5}
    assert len(ledger) == 2
    f3_row = ledger[ledger["strategy"].eq(STRATEGY_F3)].iloc[0]
    f5_row = ledger[ledger["strategy"].eq(STRATEGY_F5)].iloc[0]
    assert f3_row["blocked"] is True or f3_row["blocked"] == True
    assert f5_row["blocked"] is True or f5_row["blocked"] == True


def test_over_gate_flag_fires_on_blocked_winners() -> None:
    pool = _toy_pool(
        [
            {"symbol": "X", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": +0.04,  # blocked winner
             "candidate": "CIC2_beta_broad", "burst_count_so_far": 12},
            {"symbol": "Y", "signal_time": "2026-01-02 09:00", "entry_time": "2026-01-02 09:15",
             "exit_time": "2026-01-02 13:15", "net_return": -0.03,  # blocked loser
             "candidate": "CIC2_beta_broad", "burst_count_so_far": 12},
        ]
    )
    events = _events([("X", "S1", "2026-01-02 08:00"), ("Y", "S3", "2026-01-02 08:00")])
    ledger = _build_ledger(pool, events, ShadowConfig(cooldown_bars=8))
    f5 = ledger[ledger["strategy"].eq(STRATEGY_F5)]
    f5_blocked = f5[f5["blocked"].astype(bool)]
    over_gates = f5_blocked[f5_blocked["over_gate"].astype(bool)]
    assert len(over_gates) == 1
    assert str(over_gates.iloc[0]["symbol"]) == "X"


def test_daily_summary_groups_per_strategy_and_date() -> None:
    ledger = pd.DataFrame(
        [
            {"decision_time": pd.Timestamp("2026-01-02 09:00", tz="UTC"),
             "strategy": STRATEGY_F5, "blocked": True, "realized_net_return": -0.02, "over_gate": False},
            {"decision_time": pd.Timestamp("2026-01-02 11:00", tz="UTC"),
             "strategy": STRATEGY_F5, "blocked": True, "realized_net_return": +0.03, "over_gate": True},
            {"decision_time": pd.Timestamp("2026-01-03 09:00", tz="UTC"),
             "strategy": STRATEGY_F3, "blocked": True, "realized_net_return": -0.01, "over_gate": False},
        ]
    )
    daily = _daily_summary(ledger)
    f5_d2 = daily[(daily["date"].eq("2026-01-02")) & (daily["strategy"].eq(STRATEGY_F5))].iloc[0]
    assert int(f5_d2["blocks"]) == 2
    assert int(f5_d2["over_gate_count"]) == 1
    assert abs(f5_d2["over_gate_rate"] - 0.5) < 1e-9
    assert abs(f5_d2["gated_realized_avg"] - 0.005) < 1e-9


def test_strategy_status_reports_net_delta_and_over_gate_rate() -> None:
    ledger = pd.DataFrame(
        [
            {"decision_time": pd.Timestamp("2026-01-02", tz="UTC"), "strategy": STRATEGY_F5,
             "symbol": "X", "blocked": True, "realized_net_return": -0.02, "over_gate": False},
            {"decision_time": pd.Timestamp("2026-01-03", tz="UTC"), "strategy": STRATEGY_F5,
             "symbol": "Y", "blocked": True, "realized_net_return": -0.01, "over_gate": False},
            {"decision_time": pd.Timestamp("2026-01-04", tz="UTC"), "strategy": STRATEGY_F5,
             "symbol": "Z", "blocked": True, "realized_net_return": +0.005, "over_gate": True},
        ]
    )
    status = _strategy_status(ledger, ShadowConfig())
    f5_row = status[status["strategy"].eq(STRATEGY_F5)].iloc[0]
    assert int(f5_row["blocks"]) == 3
    assert int(f5_row["over_gate_count"]) == 1
    expected_net_delta = -(-0.02 - 0.01 + 0.005)
    assert abs(f5_row["net_delta_vs_baseline"] - expected_net_delta) < 1e-9


def test_verdict_band_holds_on_positive_net_low_over_gate() -> None:
    row = pd.Series({"blocks": 30, "gated_realized_avg": -0.015,
                     "over_gate_rate": 0.40, "net_delta_vs_baseline": 0.45})
    assert _verdict_band(row) == "shadow_holds"


def test_verdict_band_flags_over_gating_when_blocking_winners() -> None:
    row = pd.Series({"blocks": 30, "gated_realized_avg": +0.005,
                     "over_gate_rate": 0.70, "net_delta_vs_baseline": -0.15})
    assert _verdict_band(row) == "demote_over_gating"
