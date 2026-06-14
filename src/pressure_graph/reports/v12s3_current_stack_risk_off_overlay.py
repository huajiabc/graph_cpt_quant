"""v1.2s3 Current Long-Stack Risk-Off Overlay — Phase-4 finish (research only).

v1.2s proved the short failure motifs are not tradeable as shorts; v1.2s2 showed
their value as a *symbol-level risk-off gate* on the P2 max8 basket. The instruction
file's Phase-4 ask is to replay that gate against the **current** long stack — not
just P2 max8 — so the structure can be judged at its real shape:

- CIC-filtered MIR1 primary (`CIC1_beta_extreme`, primary_max=3 per v0.7D.2 paper-live)
- P2 CIC1+CIC2 max8 core basket
- O6 late-burst overflow on top of P2 max8
- MIR1 raw reference (`MIR1_reference`)
- IR2 reference (legacy paper-live — not in the v0.9D backtest cache, surfaced as
  a deferred placeholder so the table still tells a complete story)
- C2 sentinel (`CIC2_beta_broad` alone)

Decision discipline (instruction §1): **active gate = no, shadow gate = yes**. This
module emits CSVs / notes only. No paper-live or real-live wiring is touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v09b import select_portfolio
from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool, _portfolio_metrics
from pressure_graph.reports.v10d_late_burst_overflow import (
    BASELINE_MAX_POSITIONS as OVERFLOW_BASELINE_MAX,
    OverflowPolicy,
    _exposure_stats,
)
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v12s2_long_risk_off_overlay import (
    BAR_NS,
    RiskOffConfig,
    _apply_symbol_gate,
    _epoch_ns,
    stream_risk_off_events,
)


REPORT_ROOT = Path("reports/v1_2s3_current_stack_risk_off")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")
PHASE4_COOLDOWN_BARS = 48  # the v1.2s2 tuned default; we re-validate on the new stack.

# O6 overflow policy: `O6_late9_slots4_cic1_050_cic2_025` (live overflow policy from v1.0D).
O6_POLICY = OverflowPolicy(
    policy_id="O6_late9_slots4_cic1_050_cic2_025",
    min_burst_count=9,
    overflow_max_slots=4,
    cic1_size=0.50,
    cic2_size=0.25,
)


@dataclass(frozen=True)
class StackPiece:
    """One slice of the current long stack to replay the risk-off gate against."""

    key: str  # short identifier used in CSV rows and file names.
    label: str  # human-readable label.
    pool_name: str  # _focus_pool key, or sentinel for "not in cache".
    max_positions: int
    sleeve: str  # one of: "selection", "overflow", "deferred".
    notes: str = ""


CURRENT_LONG_STACK: tuple[StackPiece, ...] = (
    StackPiece(
        key="S1_CIC_FILTERED_MIR1_PRIMARY",
        label="CIC-filtered MIR1 primary (max3, paper-live)",
        pool_name="P0_CIC1_ONLY",
        max_positions=3,
        sleeve="selection",
        notes="v0.7D.2 paper-live primary; primary_max_positions=3.",
    ),
    StackPiece(
        key="S2_P2_MAX8_CORE_BASKET",
        label="P2 CIC1+CIC2 max8 core basket",
        pool_name="P2_CIC1_CIC2_COMBINED",
        max_positions=8,
        sleeve="selection",
        notes="The v1.2s2 head-line piece; replayed here for cross-validation.",
    ),
    StackPiece(
        key="S3_P2_MAX8_PLUS_O6_OVERFLOW",
        label="P2 max8 + O6 late-burst overflow",
        pool_name="P2_CIC1_CIC2_COMBINED",
        max_positions=OVERFLOW_BASELINE_MAX,
        sleeve="overflow",
        notes="Baseline=P2 max8; O6 overflow CIC1=0.50, CIC2=0.25, slots=4, min_burst=9.",
    ),
    StackPiece(
        key="S4_MIR1_RAW_REFERENCE",
        label="MIR1 raw reference",
        pool_name="P3_MIR1_RAW_WITH_CIC_LABELS",
        max_positions=5,
        sleeve="selection",
        notes="Unfiltered MIR1 — reference only, never primary in current stack.",
    ),
    StackPiece(
        key="S5_IR2_LEGACY_REFERENCE",
        label="IR2 legacy reference",
        pool_name="__NOT_IN_CACHE__",
        max_positions=3,
        sleeve="deferred",
        notes=(
            "IR2_REF lives in the v0.7D.2 paper-live signal log, not in the v0.9D "
            "backtest cache. Risk-off integration deferred until trade cache is enriched."
        ),
    ),
    StackPiece(
        key="S6_C2_SENTINEL",
        label="C2 sentinel (CIC2 broad alone)",
        pool_name="P1_CIC2_ONLY",
        max_positions=5,
        sleeve="selection",
        notes="CIC2-broad standalone — used as a sentinel on the C2 cluster gate.",
    ),
)


@dataclass(frozen=True)
class CurrentStackConfig:
    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    top_n: int = 30
    motifs: tuple[str, ...] = ("S1", "S3", "S5")
    phase4_cooldown_bars: int = PHASE4_COOLDOWN_BARS
    cooldown_sweep_bars: tuple[int, ...] = (16, 24, 32, 48, 64, 96)
    breadth_window_bars: int = 16  # carried for the v12s2 RiskOffConfig surface.
    breadth_threshold: int = 3
    burst_phase_bucket: str = "1h"  # matches v10D when simulating overflow.
    half_size_factor: float = 0.5
    pieces: tuple[StackPiece, ...] = CURRENT_LONG_STACK


def _risk_off_cfg(cfg: CurrentStackConfig, cooldown_bars: int) -> RiskOffConfig:
    return RiskOffConfig(
        report_root=cfg.report_root,
        trade_cache_path=cfg.trade_cache_path,
        top_n=cfg.top_n,
        motifs=cfg.motifs,
        symbol_cooldown_bars=cooldown_bars,
        breadth_window_bars=cfg.breadth_window_bars,
        breadth_threshold=cfg.breadth_threshold,
        half_size_factor=cfg.half_size_factor,
    )


# --------------------------------------------------------------------------------------
# Pool loading
# --------------------------------------------------------------------------------------


def _prepare_pool(trade_cache_path: Path, piece: StackPiece) -> pd.DataFrame:
    """Load a focused pool for one stack piece. Returns empty for deferred pieces."""
    if piece.sleeve == "deferred":
        return pd.DataFrame()
    trades = read_parquet(trade_cache_path)
    pool = _focus_pool(trades, piece.pool_name)
    if pool.empty:
        return pool
    pool = pool.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True, errors="coerce")
    pool = pool.dropna(subset=["signal_time"]).reset_index(drop=True)
    if "rank_first_come_first_served" not in pool.columns:
        pool["rank_first_come_first_served"] = 0.0
    if piece.sleeve == "overflow":
        # Burst-phase enrichment is needed so the O6 policy's `min_burst_count` gate
        # can fire on `burst_count_so_far`.
        pool = _add_asof_burst_phase(pool, "1h")
    return pool


# --------------------------------------------------------------------------------------
# Selection-mode replay (S1, S2, S4, S6)
# --------------------------------------------------------------------------------------


def _selection_mode_metrics(
    pool: pd.DataFrame,
    gated: pd.Series,
    piece: StackPiece,
    mode: str,
    *,
    half_size: bool = False,
    half_size_factor: float = 0.5,
) -> dict[str, object]:
    """Apply gate to a portfolio-selection piece and return one summary row."""
    if pool.empty:
        return _empty_metrics(piece, mode)
    removed = pool[gated.to_numpy()].copy()
    if half_size:
        marked = pool.copy()
        marked["risk_off_gated"] = gated.to_numpy()
        selected, skipped = select_portfolio(
            marked, score_col="rank_first_come_first_served", max_positions=piece.max_positions
        )
        if not selected.empty and "risk_off_gated" in selected.columns:
            scale = np.where(
                selected["risk_off_gated"].fillna(False).to_numpy(), half_size_factor, 1.0
            )
            selected = selected.copy()
            selected["net_return"] = pd.to_numeric(selected["net_return"], errors="coerce") * scale
    else:
        kept = pool[~gated.to_numpy()].copy()
        selected, skipped = select_portfolio(
            kept, score_col="rank_first_come_first_served", max_positions=piece.max_positions
        )
    metrics = _portfolio_metrics(
        selected,
        skipped,
        architecture="current_stack_risk_off",
        pool=piece.pool_name,
        rule=mode,
        max_positions=piece.max_positions,
        notes=piece.notes,
    )
    removed_net = pd.to_numeric(removed.get("net_return", pd.Series(dtype=float)), errors="coerce")
    dd = float(metrics.get("max_drawdown_proxy", np.nan))
    net = float(metrics.get("portfolio_net20", np.nan))
    return {
        **metrics,
        "stack_key": piece.key,
        "stack_label": piece.label,
        "sleeve": piece.sleeve,
        "mode": mode,
        "longs_gated": int(len(removed)),
        "gated_realized_net_mean": float(removed_net.mean()) if len(removed_net) else np.nan,
        "gated_loss_share": float((removed_net < 0).mean()) if len(removed_net) else np.nan,
        "return_per_drawdown": (
            float(net / abs(dd)) if dd and np.isfinite(dd) and dd != 0 else np.nan
        ),
    }


# --------------------------------------------------------------------------------------
# Overflow-mode replay (S3 — P2 max8 + O6)
# --------------------------------------------------------------------------------------


def _simulate_o6_overflow_with_gate(
    pool: pd.DataFrame,
    gated: pd.Series,
    *,
    policy: OverflowPolicy = O6_POLICY,
    half_size: bool = False,
    half_size_factor: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """O6 overflow simulator with a symbol-level risk-off gate applied at entry.

    full-skip: gated rows never enter the slot accounting; the slot is free for the
    next non-gated row (same backfill behavior as v0.9D selection).
    half-size: gated rows still consume their slot, but their weighted P&L is scaled
    by ``half_size_factor`` for both baseline and overflow sleeves.
    """
    if pool.empty:
        return pool.copy(), pool.copy()
    work = pool.copy()
    work["risk_off_gated"] = gated.to_numpy()
    active_base: list[dict[str, object]] = []
    active_overflow: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for _, row in work.sort_values(["entry_time", "symbol"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_base = [item for item in active_base if pd.Timestamp(item["exit_time"]) > entry]
        active_overflow = [
            item for item in active_overflow if pd.Timestamp(item["exit_time"]) > entry
        ]
        active_symbols = {str(item["symbol"]) for item in [*active_base, *active_overflow]}
        is_gated = bool(row.get("risk_off_gated", False))
        if str(row["symbol"]) in active_symbols:
            skipped_rows.append(
                _ledger_row(
                    row,
                    sleeve="skipped",
                    weight=0.0,
                    selection_status="skipped",
                    skip_reason="symbol_already_active",
                )
            )
            continue
        if is_gated and not half_size:
            skipped_rows.append(
                _ledger_row(
                    row,
                    sleeve="skipped",
                    weight=0.0,
                    selection_status="skipped",
                    skip_reason="risk_off_gate_full_skip",
                )
            )
            continue
        if len(active_base) < OVERFLOW_BASELINE_MAX:
            base_weight = half_size_factor if (is_gated and half_size) else 1.0
            ledger_rows.append(
                _ledger_row(row, sleeve="baseline", weight=base_weight, selection_status="selected")
            )
            active_base.append({"symbol": str(row["symbol"]), "exit_time": row["exit_time"]})
            continue
        if _overflow_allowed_local(row, policy) and len(active_overflow) < policy.overflow_max_slots:
            size = _overflow_size_local(row, policy)
            if is_gated and half_size:
                size *= half_size_factor
            ledger_rows.append(
                _ledger_row(row, sleeve="overflow", weight=size, selection_status="selected")
            )
            active_overflow.append(
                {"symbol": str(row["symbol"]), "exit_time": row["exit_time"], "weight": size}
            )
            continue
        reason = (
            "overflow_full"
            if _overflow_allowed_local(row, policy)
            else "portfolio_full_not_overflow_eligible"
        )
        skipped_rows.append(
            _ledger_row(row, sleeve="skipped", weight=0.0, selection_status="skipped", skip_reason=reason)
        )
    return pd.DataFrame(ledger_rows), pd.DataFrame(skipped_rows)


def _ledger_row(
    row: pd.Series, *, sleeve: str, weight: float, selection_status: str, skip_reason: str = ""
) -> dict[str, object]:
    payload = row.to_dict()
    payload["sleeve"] = sleeve
    payload["exposure_weight"] = float(weight)
    payload["selection_status"] = selection_status
    payload["skip_reason"] = skip_reason
    net = float(pd.to_numeric(row.get("net_return", np.nan), errors="coerce"))
    payload["weighted_return"] = net * float(weight) if np.isfinite(net) else np.nan
    return payload


def _overflow_allowed_local(row: pd.Series, policy: OverflowPolicy) -> bool:
    if policy.overflow_max_slots <= 0:
        return False
    if int(row.get("burst_count_so_far", 0)) < policy.min_burst_count:
        return False
    return _overflow_size_local(row, policy) > 0


def _overflow_size_local(row: pd.Series, policy: OverflowPolicy) -> float:
    candidate = str(row.get("candidate", ""))
    if policy.cic1_only and candidate != "CIC1_beta_extreme":
        return 0.0
    if candidate == "CIC1_beta_extreme":
        return policy.cic1_size
    if candidate == "CIC2_beta_broad":
        return policy.cic2_size
    return 0.0


def _overflow_mode_metrics(
    pool: pd.DataFrame,
    gated: pd.Series,
    piece: StackPiece,
    mode: str,
    *,
    half_size: bool = False,
    half_size_factor: float = 0.5,
) -> dict[str, object]:
    if pool.empty:
        return _empty_metrics(piece, mode)
    ledger, skipped = _simulate_o6_overflow_with_gate(
        pool, gated, policy=O6_POLICY, half_size=half_size, half_size_factor=half_size_factor
    )
    baseline_sleeve = (
        ledger[ledger["sleeve"].astype(str).eq("baseline")] if not ledger.empty else ledger
    )
    overflow_sleeve = (
        ledger[ledger["sleeve"].astype(str).eq("overflow")] if not ledger.empty else ledger
    )
    weighted = pd.to_numeric(
        ledger.get("weighted_return", pd.Series(dtype=float)), errors="coerce"
    )
    portfolio_net20 = float(weighted.sum() / OVERFLOW_BASELINE_MAX) if len(weighted) else 0.0
    equity = weighted.cumsum() / OVERFLOW_BASELINE_MAX if len(weighted) else pd.Series(dtype=float)
    dd = float((equity - equity.cummax()).min()) if len(equity) else np.nan
    exposure = _exposure_stats(ledger) if not ledger.empty else {
        "avg_exposure_units": np.nan,
        "max_exposure_units": np.nan,
    }
    overflow_gated = int(
        ledger.get("risk_off_gated", pd.Series(dtype=bool)).fillna(False).sum()
        if "risk_off_gated" in ledger.columns and not ledger.empty
        else 0
    )
    # `gated_signals_lost_to_overflow` measures Q2: how many late-burst overflow
    # signals the gate quietly disables. This is a separate count from
    # `longs_gated` (which counts pool rows the gate would suppress at signal
    # time, regardless of whether they'd have been selected by O6 anyway).
    # Q2 mechanics: count gated rows that would have actually entered the O6 sleeve
    # (positive overflow size *and* burst_count_so_far over the threshold).
    overflow_eligible = pool[gated.to_numpy()].copy()
    overflow_eligible_count = int(
        sum(_overflow_allowed_local(row, O6_POLICY) for _, row in overflow_eligible.iterrows())
    )
    base = _portfolio_metrics(
        baseline_sleeve.assign(net_return=baseline_sleeve.get("weighted_return", np.nan))
        if not baseline_sleeve.empty
        else baseline_sleeve,
        skipped.assign(net_return=skipped.get("net_return", np.nan)) if not skipped.empty else skipped,
        architecture="current_stack_risk_off",
        pool=piece.pool_name,
        rule=mode,
        max_positions=piece.max_positions,
        notes=piece.notes,
    )
    return {
        **base,
        "stack_key": piece.key,
        "stack_label": piece.label,
        "sleeve": piece.sleeve,
        "mode": mode,
        "portfolio_net20": portfolio_net20,
        "max_drawdown_proxy": dd,
        "return_per_drawdown": (
            float(portfolio_net20 / abs(dd)) if dd and np.isfinite(dd) and dd != 0 else np.nan
        ),
        "baseline_trades": int(len(baseline_sleeve)),
        "overflow_trades": int(len(overflow_sleeve)),
        "skipped_trades": int(len(skipped)),
        "longs_gated": int(gated.sum()) if not gated.empty else 0,
        "overflow_eligible_gated": overflow_eligible_count,
        "overflow_signals_gated": overflow_gated,
        "avg_exposure_units": float(exposure["avg_exposure_units"]),
        "max_exposure_units": float(exposure["max_exposure_units"]),
    }


def _empty_metrics(piece: StackPiece, mode: str) -> dict[str, object]:
    return {
        "stack_key": piece.key,
        "stack_label": piece.label,
        "sleeve": piece.sleeve,
        "mode": mode,
        "pool": piece.pool_name,
        "max_positions": piece.max_positions,
        "portfolio_net20": np.nan,
        "max_drawdown_proxy": np.nan,
        "return_per_drawdown": np.nan,
        "month_cap35_net20": np.nan,
        "max_month_contribution": np.nan,
        "max_symbol_contribution": np.nan,
        "longs_gated": 0,
        "gated_realized_net_mean": np.nan,
        "gated_loss_share": np.nan,
        "notes": piece.notes,
    }


# --------------------------------------------------------------------------------------
# Suppressed-trade attribution (Q3 + audit trail)
# --------------------------------------------------------------------------------------


def _attribute_suppressed_trades(
    pool: pd.DataFrame,
    events: pd.DataFrame,
    piece: StackPiece,
    cooldown_bars: int,
) -> pd.DataFrame:
    """For each gate-suppressed long: which motif fired, how long before, what it cost."""
    if pool.empty or events.empty:
        return pd.DataFrame()
    cfg = RiskOffConfig(symbol_cooldown_bars=cooldown_bars)
    gated = _apply_symbol_gate(pool, events, cfg)
    if not bool(gated.any()):
        return pd.DataFrame()
    window_ns = cooldown_bars * BAR_NS
    events_ns = _epoch_ns(events["feature_time"])
    by_symbol_idx: dict[str, np.ndarray] = {
        str(sym): idx.to_numpy()
        for sym, idx in events.reset_index(drop=True).groupby("symbol").groups.items()
    }
    pool_ns = _epoch_ns(pool["signal_time"])
    net_series = pd.to_numeric(
        pool.get("net_return", pd.Series([np.nan] * len(pool))), errors="coerce"
    ).to_numpy()
    candidate_series = pool.get("candidate", pd.Series([""] * len(pool))).astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for i in range(len(pool)):
        if not bool(gated.iloc[i]):
            continue
        symbol = str(pool["symbol"].iloc[i])
        candidate_event_ix = by_symbol_idx.get(symbol)
        if candidate_event_ix is None:
            continue
        signal = pool_ns[i]
        candidate_times = events_ns[candidate_event_ix]
        order = np.argsort(candidate_times)
        sorted_times = candidate_times[order]
        sorted_local_ix = candidate_event_ix[order]
        left = int(np.searchsorted(sorted_times, signal - window_ns, side="right"))
        right = int(np.searchsorted(sorted_times, signal, side="right"))
        if right <= left:
            continue
        in_window_ix = sorted_local_ix[left:right]
        # Nearest preceding motif wins the attribution row.
        latest = int(in_window_ix[-1])
        motif = str(events["motif"].iloc[latest])
        lead_ns = int(signal - events_ns[latest])
        net = float(net_series[i]) if np.isfinite(net_series[i]) else np.nan
        rows.append(
            {
                "stack_key": piece.key,
                "candidate": candidate_series[i],
                "symbol": symbol,
                "signal_time": pool["signal_time"].iloc[i],
                "feature_time_motif": events["feature_time"].iloc[latest],
                "motif": motif,
                "lead_bars": float(lead_ns / BAR_NS),
                "motif_count_in_window": int(right - left),
                "would_be_net_return": net,
                "would_be_loss": bool(np.isfinite(net) and net < 0),
                "cooldown_bars": cooldown_bars,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Cooldown sweep (Q5)
# --------------------------------------------------------------------------------------


def _sweep_one_piece(
    pool: pd.DataFrame,
    events: pd.DataFrame,
    piece: StackPiece,
    cfg: CurrentStackConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if piece.sleeve == "deferred":
        return [_empty_metrics(piece, mode="deferred")]
    if pool.empty:
        return [_empty_metrics(piece, mode="empty_pool")]
    for cooldown in cfg.cooldown_sweep_bars:
        gated = _apply_symbol_gate(pool, events, _risk_off_cfg(cfg, cooldown))
        for variant in ("full_skip", "half_size"):
            half = variant == "half_size"
            if piece.sleeve == "overflow":
                row = _overflow_mode_metrics(
                    pool,
                    gated,
                    piece,
                    mode=f"symbol_{variant}_cd{cooldown}",
                    half_size=half,
                    half_size_factor=cfg.half_size_factor,
                )
            else:
                row = _selection_mode_metrics(
                    pool,
                    gated,
                    piece,
                    mode=f"symbol_{variant}_cd{cooldown}",
                    half_size=half,
                    half_size_factor=cfg.half_size_factor,
                )
            row["cooldown_bars"] = cooldown
            row["variant"] = variant
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------------------


def _replay_piece(
    pool: pd.DataFrame,
    events: pd.DataFrame,
    piece: StackPiece,
    cfg: CurrentStackConfig,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Return baseline + symbol_risk_off rows for one piece plus its suppressed audit."""
    if piece.sleeve == "deferred" or pool.empty:
        return [_empty_metrics(piece, mode="baseline"), _empty_metrics(piece, mode="symbol_risk_off")], pd.DataFrame()
    no_gate = pd.Series(False, index=pool.index)
    gated = _apply_symbol_gate(pool, events, _risk_off_cfg(cfg, cfg.phase4_cooldown_bars))
    metrics_fn = _overflow_mode_metrics if piece.sleeve == "overflow" else _selection_mode_metrics
    rows = [
        metrics_fn(pool, no_gate, piece, mode="baseline"),
        metrics_fn(pool, gated, piece, mode="symbol_risk_off"),
        metrics_fn(
            pool,
            gated,
            piece,
            mode="symbol_half_size",
            half_size=True,
            half_size_factor=cfg.half_size_factor,
        ),
    ]
    audit = _attribute_suppressed_trades(pool, events, piece, cfg.phase4_cooldown_bars)
    return rows, audit


def _write_notes(
    report_root: Path,
    stack_summary: pd.DataFrame,
    p2_summary: pd.DataFrame,
    o6_summary: pd.DataFrame,
    suppression_audit: pd.DataFrame,
    cooldown_table: pd.DataFrame,
    cfg: CurrentStackConfig,
) -> None:
    """Answer the five instruction questions and stamp the decision verdict."""
    lines: list[str] = [
        "# v1.2s3 Current Long-Stack Risk-Off Overlay",
        "",
        "Phase-4 finish: replay the v1.2s2 symbol-level risk-off gate against the",
        "current long stack as scoped in the instruction file. Tier: research only.",
        "",
        f"- gate motifs: {', '.join(cfg.motifs)} (S2 excluded — mistimed in v1.2s)",
        f"- phase-4 cooldown: {cfg.phase4_cooldown_bars} bars (re-validated below)",
        f"- pieces replayed: {len([p for p in cfg.pieces if p.sleeve != 'deferred'])} "
        f"(+{len([p for p in cfg.pieces if p.sleeve == 'deferred'])} deferred)",
        "",
    ]

    if not stack_summary.empty:
        lines.append("## Stack-level deltas (phase-4 cooldown, full-skip vs baseline)")
        for piece in cfg.pieces:
            base = stack_summary[
                (stack_summary["stack_key"].eq(piece.key)) & (stack_summary["mode"].eq("baseline"))
            ]
            gated = stack_summary[
                (stack_summary["stack_key"].eq(piece.key))
                & (stack_summary["mode"].eq("symbol_risk_off"))
            ]
            if base.empty or gated.empty:
                lines.append(f"- {piece.key}: deferred — {piece.notes}")
                continue
            br = base.iloc[0]
            gr = gated.iloc[0]
            d_net = float(gr["portfolio_net20"]) - float(br["portfolio_net20"])
            d_dd = float(gr["max_drawdown_proxy"]) - float(br["max_drawdown_proxy"])
            lines.append(
                f"- **{piece.key}**: baseline net20={float(br['portfolio_net20']):.4%}, "
                f"max_dd={float(br['max_drawdown_proxy']):.4%}; symbol gate net20="
                f"{float(gr['portfolio_net20']):.4%} (Δ{d_net:+.4%}), max_dd="
                f"{float(gr['max_drawdown_proxy']):.4%} (Δ{d_dd:+.4%}), gated="
                f"{int(gr['longs_gated'])}, ret/dd={float(gr['return_per_drawdown']):.2f}."
            )
        lines.append("")

    lines.append("## Q1 — P2 max8 drawdown impact")
    q1_base = stack_summary[
        (stack_summary["stack_key"].eq("S2_P2_MAX8_CORE_BASKET"))
        & (stack_summary["mode"].eq("baseline"))
    ]
    q1_gate = stack_summary[
        (stack_summary["stack_key"].eq("S2_P2_MAX8_CORE_BASKET"))
        & (stack_summary["mode"].eq("symbol_risk_off"))
    ]
    if not q1_base.empty and not q1_gate.empty:
        base_dd = float(q1_base.iloc[0]["max_drawdown_proxy"])
        gate_dd = float(q1_gate.iloc[0]["max_drawdown_proxy"])
        improved = abs(gate_dd) < abs(base_dd)
        lines.append(
            f"- P2 max8 baseline max_dd={base_dd:.4%}; symbol-gate max_dd={gate_dd:.4%}; "
            f"{'improved' if improved else 'worse-or-flat'} (Δ={gate_dd - base_dd:+.4%})."
        )
    lines.append("")

    lines.append("## Q2 — O6 late-burst overflow false kills")
    q2_base = stack_summary[
        (stack_summary["stack_key"].eq("S3_P2_MAX8_PLUS_O6_OVERFLOW"))
        & (stack_summary["mode"].eq("baseline"))
    ]
    q2_gate = stack_summary[
        (stack_summary["stack_key"].eq("S3_P2_MAX8_PLUS_O6_OVERFLOW"))
        & (stack_summary["mode"].eq("symbol_risk_off"))
    ]
    if not q2_gate.empty and not q2_base.empty:
        base_r = q2_base.iloc[0]
        r = q2_gate.iloc[0]
        overflow_eligible_gated = int(r.get("overflow_eligible_gated", 0))
        overflow_trades_baseline = int(base_r.get("overflow_trades", 0))
        overflow_trades_after = int(r.get("overflow_trades", 0))
        delta_net = float(r["portfolio_net20"]) - float(base_r["portfolio_net20"])
        lines.append(
            f"- overflow-eligible signals the gate suppressed: {overflow_eligible_gated} "
            f"(overflow sleeve: {overflow_trades_baseline} baseline → {overflow_trades_after} gated). "
            f"Net20 delta vs P2+O6 baseline: {delta_net:+.4%}."
        )
    lines.append("")

    lines.append("## Q3 — Suppressed-trade attribution")
    if not suppression_audit.empty:
        loss_share = float(suppression_audit["would_be_loss"].mean())
        avg_net = float(
            pd.to_numeric(suppression_audit["would_be_net_return"], errors="coerce").mean()
        )
        n = int(len(suppression_audit))
        by_motif = (
            suppression_audit.groupby("motif")["would_be_net_return"].agg(["count", "mean"]).reset_index()
        )
        lines.append(
            f"- {n} longs suppressed across the stack. would-be loss share="
            f"{loss_share:.2%}, would-be avg net20={avg_net:.4%}."
        )
        for _, row in by_motif.iterrows():
            lines.append(
                f"  - motif {row['motif']}: {int(row['count'])} suppressions, "
                f"avg would-be net20={float(row['mean']):.4%}."
            )
    else:
        lines.append("- no suppressed rows (event stream empty or pool empty).")
    lines.append("")

    lines.append("## Q4 — Full-skip vs size-down")
    if not cooldown_table.empty:
        at_phase4 = cooldown_table[cooldown_table["cooldown_bars"].eq(cfg.phase4_cooldown_bars)]
        if not at_phase4.empty:
            for piece in cfg.pieces:
                rows = at_phase4[at_phase4["stack_key"].eq(piece.key)]
                if rows.empty:
                    continue
                full = rows[rows["variant"].eq("full_skip")]
                half = rows[rows["variant"].eq("half_size")]
                if full.empty or half.empty:
                    continue
                lines.append(
                    f"- {piece.key}: full-skip net20={float(full.iloc[0]['portfolio_net20']):.4%}, "
                    f"max_dd={float(full.iloc[0]['max_drawdown_proxy']):.4%}; "
                    f"half-size net20={float(half.iloc[0]['portfolio_net20']):.4%}, "
                    f"max_dd={float(half.iloc[0]['max_drawdown_proxy']):.4%}."
                )
    lines.append("")

    lines.append("## Q5 — Is the 48-bar cooldown still right?")
    if not cooldown_table.empty:
        full = cooldown_table[cooldown_table["variant"].eq("full_skip")].copy()
        for piece in cfg.pieces:
            rows = full[full["stack_key"].eq(piece.key)]
            if rows.empty:
                continue
            best = rows.sort_values("return_per_drawdown", ascending=False).iloc[0]
            lines.append(
                f"- {piece.key}: best cooldown={int(best['cooldown_bars'])} bars, "
                f"net20={float(best['portfolio_net20']):.4%}, max_dd="
                f"{float(best['max_drawdown_proxy']):.4%}, ret/dd="
                f"{float(best['return_per_drawdown']):.2f}."
            )
    lines.append("")

    lines.extend(
        [
            "## Verdict",
            "- active gate: **no** — Phase-4 stays off the live execution path.",
            "- shadow gate: **yes** — feed the gate's decisions into `risk_off_signal_shadow.csv`",
            "  alongside the existing v1.2s2 wiring; let it accrue evidence on the *current* stack.",
            "",
            "## Discipline",
            "- Strict as-of: a long is only gated by confirmations with feature_time <= its signal_time.",
            "- IR2 reference is deferred — it lives in the paper-live signal log, not the v0.9D cache.",
            "- No shadow / paper-live / real-live permission changes.",
        ]
    )

    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v12s3_current_stack_risk_off(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: CurrentStackConfig = CurrentStackConfig(),
) -> dict[str, Path]:
    """Produce the six instruction-mandated CSVs + the candidate_notes.md verdict."""
    report_root = ensure_dir(cfg.report_root)
    if not cfg.trade_cache_path.exists():
        raise FileNotFoundError(
            f"long trade cache not found: {cfg.trade_cache_path} (run run-v09d first)"
        )

    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n][
            "symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    events = stream_risk_off_events(
        feature_path,
        rank30,
        rank90,
        symbols,
        config,
        _risk_off_cfg(cfg, cfg.phase4_cooldown_bars),
    )

    all_rows: list[dict[str, object]] = []
    suppression_frames: list[pd.DataFrame] = []
    cooldown_rows: list[dict[str, object]] = []

    for piece in cfg.pieces:
        pool = _prepare_pool(cfg.trade_cache_path, piece)
        rows, audit = _replay_piece(pool, events, piece, cfg)
        all_rows.extend(rows)
        if not audit.empty:
            suppression_frames.append(audit)
        cooldown_rows.extend(_sweep_one_piece(pool, events, piece, cfg))

    stack_summary = pd.DataFrame(all_rows)
    cooldown_table = pd.DataFrame(cooldown_rows)
    suppression_audit = (
        pd.concat(suppression_frames, ignore_index=True) if suppression_frames else pd.DataFrame()
    )

    p2_summary = stack_summary[
        stack_summary["stack_key"].eq("S2_P2_MAX8_CORE_BASKET")
    ].reset_index(drop=True)
    o6_summary = stack_summary[
        stack_summary["stack_key"].eq("S3_P2_MAX8_PLUS_O6_OVERFLOW")
    ].reset_index(drop=True)
    cooldown_validation = cooldown_table.copy()

    outputs = {
        "risk_off_on_current_long_stack": report_root / "risk_off_on_current_long_stack.csv",
        "risk_off_on_p2_max8": report_root / "risk_off_on_p2_max8.csv",
        "risk_off_on_p2_max8_plus_o6": report_root / "risk_off_on_p2_max8_plus_o6.csv",
        "suppressed_trade_attribution": report_root / "suppressed_trade_attribution.csv",
        "cooldown_48bar_validation": report_root / "cooldown_48bar_validation.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    stack_summary.to_csv(outputs["risk_off_on_current_long_stack"], index=False)
    p2_summary.to_csv(outputs["risk_off_on_p2_max8"], index=False)
    o6_summary.to_csv(outputs["risk_off_on_p2_max8_plus_o6"], index=False)
    suppression_audit.to_csv(outputs["suppressed_trade_attribution"], index=False)
    cooldown_validation.to_csv(outputs["cooldown_48bar_validation"], index=False)
    _write_notes(
        report_root,
        stack_summary,
        p2_summary,
        o6_summary,
        suppression_audit,
        cooldown_validation,
        cfg,
    )
    return outputs


__all__ = [
    "CURRENT_LONG_STACK",
    "CurrentStackConfig",
    "O6_POLICY",
    "REPORT_ROOT",
    "StackPiece",
    "_attribute_suppressed_trades",
    "_overflow_mode_metrics",
    "_selection_mode_metrics",
    "_simulate_o6_overflow_with_gate",
    "write_v12s3_current_stack_risk_off",
]
