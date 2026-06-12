from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v12s3_current_stack_risk_off_overlay import (
    CURRENT_LONG_STACK,
    CurrentStackConfig,
    O6_POLICY,
    StackPiece,
    _attribute_suppressed_trades,
    _empty_metrics,
    _overflow_mode_metrics,
    _replay_piece,
    _selection_mode_metrics,
    _simulate_o6_overflow_with_gate,
    _sweep_one_piece,
)


def _selection_pool(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["rank_first_come_first_served"] = 0.0
    if "month" not in df.columns:
        df["month"] = df["entry_time"].dt.strftime("%Y-%m")
    return df


def _overflow_pool() -> pd.DataFrame:
    # 10 rows, last two are late-burst CIC1 candidates that pass the O6 min_burst_count=9 gate.
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for idx in range(10):
        late = idx >= 8
        entry = base + pd.Timedelta(minutes=5 * idx)
        rows.append(
            {
                "row_id": idx,
                "exchange": "bybit",
                "symbol": f"S{idx}USDT",
                "candidate": "CIC1_beta_extreme" if late else "CIC2_beta_broad",
                "signal_time": entry,
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "cost_single_side_bps": 20.0,
                "net_return": 0.04 if late else 0.01,
                "month": "2026-01",
                "holding_minutes": 240.0,
            }
        )
    df = pd.DataFrame(rows)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    return _add_asof_burst_phase(df, "1h")


def test_current_long_stack_covers_all_six_instruction_pieces() -> None:
    keys = [p.key for p in CURRENT_LONG_STACK]
    assert keys == [
        "S1_CIC_FILTERED_MIR1_PRIMARY",
        "S2_P2_MAX8_CORE_BASKET",
        "S3_P2_MAX8_PLUS_O6_OVERFLOW",
        "S4_MIR1_RAW_REFERENCE",
        "S5_IR2_LEGACY_REFERENCE",
        "S6_C2_SENTINEL",
    ]
    # IR2 is documented as deferred; everything else has a cacheable pool.
    deferred = [p for p in CURRENT_LONG_STACK if p.sleeve == "deferred"]
    assert [p.key for p in deferred] == ["S5_IR2_LEGACY_REFERENCE"]


def test_selection_mode_drops_gated_long_and_records_attribution() -> None:
    pool = _selection_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-01 10:00",
                "entry_time": "2026-01-01 10:15",
                "exit_time": "2026-01-01 14:00",
                "net_return": -0.05,
                "holding_minutes": 225.0,
                "candidate": "CIC1_beta_extreme",
            },
            {
                "symbol": "BBBUSDT",
                "signal_time": "2026-01-02 10:00",
                "entry_time": "2026-01-02 10:15",
                "exit_time": "2026-01-02 14:00",
                "net_return": 0.04,
                "holding_minutes": 225.0,
                "candidate": "CIC1_beta_extreme",
            },
        ]
    )
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S1_CIC_FILTERED_MIR1_PRIMARY")
    gated = pd.Series([True, False])
    metrics = _selection_mode_metrics(pool, gated, piece, mode="symbol_risk_off")
    assert metrics["longs_gated"] == 1
    # The removed long was a loser; the gate's gated_realized_net_mean reflects that.
    assert metrics["gated_realized_net_mean"] == -0.05
    assert metrics["gated_loss_share"] == 1.0
    assert metrics["sleeve"] == "selection"


def test_selection_mode_half_size_actually_scales_kept_gated_long() -> None:
    """Regression: the underscore-prefix gate marker column was silently dropped by itertuples()."""
    pool = _selection_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-01 10:00",
                "entry_time": "2026-01-01 10:15",
                "exit_time": "2026-01-01 14:00",
                "net_return": 0.08,
                "holding_minutes": 225.0,
                "candidate": "CIC1_beta_extreme",
            }
        ]
    )
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S1_CIC_FILTERED_MIR1_PRIMARY")
    baseline = _selection_mode_metrics(pool, pd.Series([False]), piece, mode="baseline")
    half = _selection_mode_metrics(
        pool, pd.Series([True]), piece, mode="symbol_half_size", half_size=True, half_size_factor=0.5
    )
    assert half["longs_gated"] == 1
    assert abs(half["portfolio_net20"] - 0.5 * baseline["portfolio_net20"]) < 1e-9


def test_overflow_simulator_skips_gated_rows_in_full_skip_mode() -> None:
    pool = _overflow_pool()
    gated = pd.Series([False] * 8 + [True, True])  # both late overflow rows gated
    ledger, skipped = _simulate_o6_overflow_with_gate(pool, gated, policy=O6_POLICY)
    # All 8 baseline rows fill the base sleeve; the two gated overflow rows are skipped.
    assert int(ledger["sleeve"].eq("baseline").sum()) == 8
    assert int(ledger["sleeve"].eq("overflow").sum()) == 0
    assert int(skipped["skip_reason"].eq("risk_off_gate_full_skip").sum()) == 2


def test_overflow_simulator_half_size_keeps_overflow_at_reduced_weight() -> None:
    pool = _overflow_pool()
    gated = pd.Series([False] * 8 + [True, True])
    ledger, _ = _simulate_o6_overflow_with_gate(
        pool, gated, policy=O6_POLICY, half_size=True, half_size_factor=0.5
    )
    overflow = ledger[ledger["sleeve"].eq("overflow")]
    assert len(overflow) == 2
    # CIC1 base size 0.50 * half_size 0.5 = 0.25
    weights = overflow["exposure_weight"].astype(float).tolist()
    assert all(abs(w - 0.25) < 1e-9 for w in weights), weights


def test_overflow_mode_metrics_counts_overflow_eligible_gated() -> None:
    pool = _overflow_pool()
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S3_P2_MAX8_PLUS_O6_OVERFLOW")
    gated = pd.Series([False] * 8 + [True, True])
    metrics = _overflow_mode_metrics(pool, gated, piece, mode="symbol_risk_off")
    assert metrics["longs_gated"] == 2
    assert metrics["overflow_eligible_gated"] == 2  # both gated rows were CIC1, O6-eligible.
    assert metrics["overflow_trades"] == 0  # gate skipped them out.
    assert metrics["sleeve"] == "overflow"


def test_attribute_suppressed_trades_carries_motif_and_lead_bars() -> None:
    pool = _selection_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-01 10:00",
                "entry_time": "2026-01-01 10:15",
                "exit_time": "2026-01-01 14:00",
                "net_return": -0.02,
                "holding_minutes": 225.0,
                "candidate": "CIC1_beta_extreme",
            },
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-05 10:00",
                "entry_time": "2026-01-05 10:15",
                "exit_time": "2026-01-05 14:00",
                "net_return": 0.03,
                "holding_minutes": 225.0,
                "candidate": "CIC1_beta_extreme",
            },
        ]
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAAUSDT", "AAAUSDT"],
            "motif": ["S5", "S1"],
            "feature_time": pd.to_datetime(["2026-01-01 06:00", "2026-01-01 09:00"], utc=True),
        }
    )
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S1_CIC_FILTERED_MIR1_PRIMARY")
    audit = _attribute_suppressed_trades(pool, events, piece, cooldown_bars=32)
    assert len(audit) == 1  # only the first long is within the 8h cooldown
    row = audit.iloc[0]
    assert row["motif"] == "S1"  # nearest-preceding motif wins attribution
    assert row["lead_bars"] == 4.0  # 1h apart = 4 * 15m bars
    assert bool(row["would_be_loss"]) is True
    assert row["motif_count_in_window"] == 2


def test_attribute_suppressed_trades_is_strict_as_of() -> None:
    pool = _selection_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-01 10:00",
                "entry_time": "2026-01-01 10:15",
                "exit_time": "2026-01-01 14:00",
                "net_return": 0.01,
                "holding_minutes": 225.0,
                "candidate": "CIC1_beta_extreme",
            }
        ]
    )
    # Motif fires AFTER the signal — must never attribute, no audit row.
    events = pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "motif": ["S1"],
            "feature_time": pd.to_datetime(["2026-01-01 11:00"], utc=True),
        }
    )
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S1_CIC_FILTERED_MIR1_PRIMARY")
    audit = _attribute_suppressed_trades(pool, events, piece, cooldown_bars=32)
    assert audit.empty


def test_sweep_one_piece_emits_both_variants_per_cooldown() -> None:
    pool = _selection_pool(
        [
            {
                "symbol": "AAAUSDT",
                "signal_time": "2026-01-01 10:00",
                "entry_time": "2026-01-01 10:15",
                "exit_time": "2026-01-01 14:00",
                "net_return": 0.02,
                "holding_minutes": 225.0,
                "candidate": "CIC1_beta_extreme",
            }
        ]
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "motif": ["S1"],
            "feature_time": pd.to_datetime(["2026-01-01 08:00"], utc=True),
        }
    )
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S1_CIC_FILTERED_MIR1_PRIMARY")
    cfg = CurrentStackConfig(cooldown_sweep_bars=(16, 48))
    rows = _sweep_one_piece(pool, events, piece, cfg)
    cd_values = sorted({int(r["cooldown_bars"]) for r in rows})
    variants = sorted({str(r["variant"]) for r in rows})
    assert cd_values == [16, 48]
    assert variants == ["full_skip", "half_size"]
    assert len(rows) == 4  # 2 cooldowns × 2 variants


def test_deferred_piece_emits_placeholder_without_crash() -> None:
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S5_IR2_LEGACY_REFERENCE")
    pool = pd.DataFrame()  # deferred pieces never have a pool.
    rows, audit = _replay_piece(pool, pd.DataFrame(), piece, CurrentStackConfig())
    assert audit.empty
    assert {r["mode"] for r in rows} == {"baseline", "symbol_risk_off"}
    assert all(r["stack_key"] == "S5_IR2_LEGACY_REFERENCE" for r in rows)
    assert all(r["sleeve"] == "deferred" for r in rows)
    assert all(np.isnan(r["portfolio_net20"]) for r in rows)


def test_empty_metrics_keeps_stack_identity_columns() -> None:
    piece = next(p for p in CURRENT_LONG_STACK if p.key == "S2_P2_MAX8_CORE_BASKET")
    m = _empty_metrics(piece, mode="empty_pool")
    assert m["stack_key"] == "S2_P2_MAX8_CORE_BASKET"
    assert m["mode"] == "empty_pool"
    assert m["sleeve"] == "selection"
    assert m["max_positions"] == 8
    assert np.isnan(m["portfolio_net20"])
