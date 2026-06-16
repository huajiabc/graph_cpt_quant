from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v3_5_failure_risk_layer_bridge import (
    ACTION_ALLOW,
    ACTION_FLAG_PROTECT_A,
    ACTION_SKIP_FULL,
    ACTION_SKIP_OVERFLOW,
    BASELINES,
    FAILURE_ACTIONS,
    V35Config,
    _build_decisions,
    _per_row_failure_recent,
    _run_cell,
    _simulate_b0_selection,
    _simulate_b1_overflow,
    _simulate_b3_protect_a_cap,
)


def _toy_pool(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["rank_first_come_first_served"] = 0.0
    if "candidate" not in df.columns:
        df["candidate"] = "CIC1_beta_extreme"
    if "cp60_would_exit" not in df.columns:
        df["cp60_would_exit"] = False
    if "protect_a_active" not in df.columns:
        df["protect_a_active"] = False
    if "month" not in df.columns:
        df["month"] = df["entry_time"].dt.strftime("%Y-%m")
    df["candidate"] = df["candidate"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce").fillna(0.0)
    return _add_asof_burst_phase(df, "1h")


def _events(records: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": symbol, "motif": motif, "feature_time": pd.Timestamp(ts, tz="UTC")}
            for symbol, motif, ts in records
        ]
    )


def test_actions_and_baselines_form_the_instructed_matrix() -> None:
    assert tuple(a.code for a in FAILURE_ACTIONS) == ("F0", "F1", "F2", "F3", "F4", "F5")
    assert tuple(b.code for b in BASELINES) == ("B0", "B1", "B2", "B3")
    assert FAILURE_ACTIONS[0].channel == "off"
    f2 = next(a for a in FAILURE_ACTIONS if a.code == "F2")
    assert f2.motifs == ("S1",)
    f3 = next(a for a in FAILURE_ACTIONS if a.code == "F3")
    assert f3.channel == "symbol_overflow_only"
    f5 = next(a for a in FAILURE_ACTIONS if a.code == "F5")
    assert f5.channel == "symbol_cic2_only"


def test_failure_recent_is_strict_as_of() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-01 09:00",
                "entry_time": "2026-01-01 09:15",
                "exit_time": "2026-01-01 13:15",
                "net_return": 0.02,
            },
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.01,
            },
        ]
    )
    events = _events([("AAAUSDT", "S1", "2026-01-01 12:00")])
    flag = _per_row_failure_recent(pool, events, cooldown_bars=48)
    # row 0 signal_time precedes the event -> as-of False
    # row 1 signal_time is 21h after the event, inside the 12h cooldown? 48 * 15m = 12h. 21h > 12h -> False
    assert list(flag) == [False, False]
    flag48 = _per_row_failure_recent(pool, events, cooldown_bars=96)  # 24h
    # row 1 is within 24h of the event now -> True; row 0 still precedes.
    assert list(flag48) == [False, True]


def test_f0_record_only_allows_everything() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": 0.02,
            }
        ]
    )
    events = _events([("AAAUSDT", "S1", "2026-01-02 08:00")])
    cfg = V35Config(cooldown_bars=8)
    f0 = next(a for a in FAILURE_ACTIONS if a.code == "F0")
    decisions = _build_decisions(pool, f0, events, cfg)
    assert (decisions == ACTION_ALLOW).all()


def test_f1_skip_full_on_failure_recent() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.02,
            },
            {
                "symbol": "BBBUSDT",
                "signal_time": "2026-01-02 10:00",
                "entry_time": "2026-01-02 10:15",
                "exit_time": "2026-01-02 14:15",
                "net_return": 0.01,
            },
        ]
    )
    # AAAUSDT failure event 1h before its signal -> within 8-bar (2h) cooldown
    events = _events([("AAAUSDT", "S1", "2026-01-02 08:00")])
    cfg = V35Config(cooldown_bars=8)
    f1 = next(a for a in FAILURE_ACTIONS if a.code == "F1")
    decisions = _build_decisions(pool, f1, events, cfg)
    assert decisions.iloc[0] == ACTION_SKIP_FULL
    assert decisions.iloc[1] == ACTION_ALLOW


def test_f3_skip_overflow_only() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.02,
            }
        ]
    )
    events = _events([("AAAUSDT", "S1", "2026-01-02 08:00")])
    cfg = V35Config(cooldown_bars=8)
    f3 = next(a for a in FAILURE_ACTIONS if a.code == "F3")
    decisions = _build_decisions(pool, f3, events, cfg)
    assert decisions.iloc[0] == ACTION_SKIP_OVERFLOW


def test_f4_only_fires_when_protect_a_active() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.02,
                "protect_a_active": True,
            },
            {
                "symbol": "BBBUSDT",
                "signal_time": "2026-01-02 10:00",
                "entry_time": "2026-01-02 10:15",
                "exit_time": "2026-01-02 14:15",
                "net_return": -0.01,
                "protect_a_active": False,
            },
        ]
    )
    events = _events([
        ("AAAUSDT", "S1", "2026-01-02 08:00"),
        ("BBBUSDT", "S1", "2026-01-02 09:00"),
    ])
    cfg = V35Config(cooldown_bars=8)
    f4 = next(a for a in FAILURE_ACTIONS if a.code == "F4")
    decisions = _build_decisions(pool, f4, events, cfg)
    assert decisions.iloc[0] == ACTION_FLAG_PROTECT_A
    assert decisions.iloc[1] == ACTION_ALLOW  # protect_a_active is False


def test_f5_only_fires_on_cic2() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.02,
                "candidate": "CIC1_beta_extreme",
            },
            {
                "symbol": "BBBUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.02,
                "candidate": "CIC2_beta_broad",
            },
        ]
    )
    events = _events([
        ("AAAUSDT", "S1", "2026-01-02 08:00"),
        ("BBBUSDT", "S1", "2026-01-02 08:00"),
    ])
    cfg = V35Config(cooldown_bars=8)
    f5 = next(a for a in FAILURE_ACTIONS if a.code == "F5")
    decisions = _build_decisions(pool, f5, events, cfg)
    assert decisions.iloc[0] == ACTION_ALLOW
    assert decisions.iloc[1] == ACTION_SKIP_FULL


def test_b0_simulator_drops_skip_full_rows() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": f"S{i}USDT",
                "signal_time": f"2026-01-02 0{i}:00",
                "entry_time": f"2026-01-02 0{i}:15",
                "exit_time": f"2026-01-02 0{i+1}:15",
                "net_return": 0.01 if i % 2 == 0 else -0.01,
            }
            for i in range(6)
        ]
    )
    decisions = pd.Series([ACTION_ALLOW] * 6, index=pool.index)
    decisions.iloc[3] = ACTION_SKIP_FULL
    decisions.iloc[5] = ACTION_SKIP_OVERFLOW  # no-op at B0
    selected, skipped = _simulate_b0_selection(pool, decisions, max_positions=8)
    assert len(selected) == 5  # one row removed
    assert "S3USDT" not in selected["symbol"].astype(str).tolist()
    assert (skipped["skip_reason"].astype(str) == "risk_off_gate_full_skip").any()


def test_b1_simulator_routes_skip_overflow_correctly() -> None:
    rows = []
    base = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    for i in range(12):
        rows.append(
            {
                "symbol": f"S{i}USDT",
                "signal_time": base + pd.Timedelta(minutes=15 * i),
                "entry_time": base + pd.Timedelta(minutes=15 * i + 5),
                "exit_time": base + pd.Timedelta(minutes=15 * i + 5) + pd.Timedelta(hours=4),
                "net_return": 0.02,
                "candidate": "CIC1_beta_extreme",
            }
        )
    pool = _toy_pool(rows)
    # Force burst_count high enough to allow O6 entries
    pool["burst_count_so_far"] = 12
    decisions = pd.Series([ACTION_ALLOW] * len(pool), index=pool.index)
    decisions.iloc[10] = ACTION_SKIP_OVERFLOW
    ledger, skipped = _simulate_b1_overflow(pool, decisions, policy=V35Config().overflow_policy)
    # First 8 → baseline; rest may go overflow up to 4 slots
    sleeves = ledger["sleeve"].astype(str).tolist() if not ledger.empty else []
    assert sleeves[:8] == ["baseline"] * 8
    # Row 10's skip_overflow must NOT appear in overflow sleeve
    assert "S10USDT" not in {str(s) for s in ledger.get("symbol", pd.Series(dtype=str))}
    assert any(str(r) == "risk_off_gate_overflow_only" for r in skipped["skip_reason"]) if not skipped.empty else False


def test_b3_protect_a_cap_blocks_third_protected_long() -> None:
    rows = []
    base = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    for i in range(5):
        rows.append(
            {
                "symbol": f"P{i}USDT",
                "signal_time": base + pd.Timedelta(minutes=15 * i),
                "entry_time": base + pd.Timedelta(minutes=15 * i + 5),
                "exit_time": base + pd.Timedelta(minutes=15 * i + 5) + pd.Timedelta(hours=6),
                "net_return": 0.02,
                "candidate": "CIC1_beta_extreme",
                "protect_a_active": True,
            }
        )
    pool = _toy_pool(rows)
    decisions = pd.Series([ACTION_ALLOW] * len(pool), index=pool.index)
    ledger, skipped = _simulate_b3_protect_a_cap(
        pool, decisions, policy=V35Config().overflow_policy, protect_a_cap=2
    )
    # Only first 2 protected longs admitted into baseline; rest skipped due to cap
    accepted = ledger["symbol"].astype(str).tolist() if not ledger.empty else []
    assert accepted == ["P0USDT", "P1USDT"]
    assert all(
        str(r) == "protect_a_cap_reached"
        for r in skipped["skip_reason"].astype(str)
    )


def test_run_cell_b0_f1_pipeline_end_to_end() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.02,
            },
            {
                "symbol": "BBBUSDT",
                "signal_time": "2026-01-02 10:00",
                "entry_time": "2026-01-02 10:15",
                "exit_time": "2026-01-02 14:15",
                "net_return": 0.01,
            },
        ]
    )
    events = _events([("AAAUSDT", "S1", "2026-01-02 08:00")])
    f1 = next(a for a in FAILURE_ACTIONS if a.code == "F1")
    b0 = next(b for b in BASELINES if b.code == "B0")
    cfg = V35Config(cooldown_bars=8)
    ledger, skipped, decisions = _run_cell(pool, f1, b0, events, cfg)
    assert len(ledger) == 1
    assert ledger.iloc[0]["symbol"] == "BBBUSDT"
    assert decisions.iloc[0] == ACTION_SKIP_FULL


def test_b2_cp60_prefilter_drops_weak_stagnant_rows() -> None:
    pool = _toy_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-02 09:00",
                "entry_time": "2026-01-02 09:15",
                "exit_time": "2026-01-02 13:15",
                "net_return": -0.01,
                "cp60_would_exit": True,
            },
            {
                "symbol": "BBBUSDT",
                "signal_time": "2026-01-02 10:00",
                "entry_time": "2026-01-02 10:15",
                "exit_time": "2026-01-02 14:15",
                "net_return": 0.02,
                "cp60_would_exit": False,
            },
        ]
    )
    events = pd.DataFrame(columns=["symbol", "motif", "feature_time"])
    f0 = next(a for a in FAILURE_ACTIONS if a.code == "F0")
    b2 = next(b for b in BASELINES if b.code == "B2")
    cfg = V35Config(cooldown_bars=8)
    ledger, skipped, decisions = _run_cell(pool, f0, b2, events, cfg)
    selected_symbols = (
        ledger["symbol"].astype(str).tolist() if not ledger.empty else []
    )
    assert "AAAUSDT" not in selected_symbols
    assert "BBBUSDT" in selected_symbols
