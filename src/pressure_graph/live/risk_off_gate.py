"""Reusable long risk-off gate — the portable form of the v1.2s2 result.

v1.2s2 showed that a symbol-level risk-off gate (stand aside on a name shortly
after its own failure motif) improves the CIC long book's drawdown and
risk-adjusted return. This module packages that gate so the live long pipeline
can consult it, and provides a *behavior-preserving* shadow annotation: it never
removes a live signal, it only records whether the gate WOULD have suppressed it,
so paper-live samples accrue the gate's would-be decisions for validation.

Single-point API for a live loop:
    suppressed, reason = risk_off_decision(symbol, decision_time, events, cfg)

Strict as-of: a decision at time T is only informed by failure confirmations with
feature_time <= T. Nothing here changes paper-live or real-live behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v12s2_long_risk_off_overlay import BAR_NS, _epoch_ns
from pressure_graph.reports.v12s_short_motif_atlas import DETECTORS, MOTIF_PARAMS

# Long-entry gate columns produced by the v07d2 live prep (add_v07d2_live_columns).
LONG_GATE_COLUMNS = {
    "MIR1_reference": "c2_mir1_raw",
    "CIC1_beta_extreme": "c2_bucket_beta_extreme_overextended",
    "CIC2_beta_broad": "c2_beta_continuation",
}


@dataclass(frozen=True)
class RiskOffGateConfig:
    motifs: tuple[str, ...] = ("S1", "S3", "S5")
    symbol_cooldown_bars: int = 48  # 12h, the v1.2s3 validated shadow default.
    top_n: int = 30


def build_risk_off_events(prepared: pd.DataFrame, cfg: RiskOffGateConfig = RiskOffGateConfig()) -> pd.DataFrame:
    """Failure-motif confirmations from an in-memory prepared feature frame.

    Detectors degrade gracefully: a motif whose input columns are absent simply
    produces no events (the _f/_b helpers return NaN/False), so this is safe on
    live frames that may not carry every feature.
    """
    if prepared.empty or "symbol" not in prepared.columns:
        return pd.DataFrame(columns=["symbol", "motif", "feature_time"])
    rows: list[dict[str, object]] = []
    for symbol, group in prepared.groupby("symbol", sort=False):
        if "dynamic_all_rank" in group.columns:
            group = group[pd.to_numeric(group["dynamic_all_rank"], errors="coerce") <= cfg.top_n]
        if group.empty:
            continue
        group = group.sort_values("bar_open_time").reset_index(drop=True)
        feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
        for motif_code in cfg.motifs:
            for confirmation_idx, _ in DETECTORS[motif_code](group, MOTIF_PARAMS[motif_code]):
                ts = feature_time.iloc[confirmation_idx]
                if pd.notna(ts):
                    rows.append({"symbol": str(symbol), "motif": motif_code, "feature_time": ts})
    events = pd.DataFrame(rows, columns=["symbol", "motif", "feature_time"])
    return events.sort_values("feature_time").reset_index(drop=True) if not events.empty else events


def _events_by_symbol(events: pd.DataFrame) -> dict[str, np.ndarray]:
    if events.empty:
        return {}
    out: dict[str, np.ndarray] = {}
    for sym, grp in events.groupby("symbol"):
        out[str(sym)] = np.sort(_epoch_ns(grp["feature_time"]))
    return out


def risk_off_decision(
    symbol: str,
    decision_time: pd.Timestamp,
    events: pd.DataFrame,
    cfg: RiskOffGateConfig = RiskOffGateConfig(),
    *,
    _by_symbol: dict[str, np.ndarray] | None = None,
) -> tuple[bool, str]:
    """Would a new long on ``symbol`` at ``decision_time`` be risk-off suppressed?"""
    by_symbol = _by_symbol if _by_symbol is not None else _events_by_symbol(events)
    times = by_symbol.get(str(symbol))
    if times is None or len(times) == 0:
        return False, ""
    decision_ns = int(pd.Timestamp(decision_time).tz_convert("UTC").value) if pd.Timestamp(decision_time).tzinfo else int(pd.Timestamp(decision_time, tz="UTC").value)
    window_ns = cfg.symbol_cooldown_bars * BAR_NS
    left = np.searchsorted(times, decision_ns - window_ns, side="right")
    right = np.searchsorted(times, decision_ns, side="right")
    if right > left:
        return True, f"recent_failure_within_{cfg.symbol_cooldown_bars}_bars"
    return False, ""


def annotate_signals_with_risk_off(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    cfg: RiskOffGateConfig = RiskOffGateConfig(),
    *,
    decision_col: str = "signal_time",
) -> pd.DataFrame:
    """Add risk_off_suppressed / risk_off_reason columns. Pure, behavior-preserving."""
    out = signals.copy()
    if out.empty:
        out["risk_off_suppressed"] = pd.Series(dtype=bool)
        out["risk_off_reason"] = pd.Series(dtype=str)
        out["risk_off_cooldown_bars"] = pd.Series(dtype=int)
        return out
    by_symbol = _events_by_symbol(events)
    decision_ns = _epoch_ns(out[decision_col])
    window_ns = cfg.symbol_cooldown_bars * BAR_NS
    symbols = out["symbol"].astype(str).to_numpy()
    suppressed = np.zeros(len(out), dtype=bool)
    for i in range(len(out)):
        times = by_symbol.get(symbols[i])
        if times is None:
            continue
        left = np.searchsorted(times, decision_ns[i] - window_ns, side="right")
        right = np.searchsorted(times, decision_ns[i], side="right")
        if right > left:
            suppressed[i] = True
    out["risk_off_suppressed"] = suppressed
    out["risk_off_reason"] = np.where(suppressed, f"recent_failure_within_{cfg.symbol_cooldown_bars}_bars", "")
    out["risk_off_cooldown_bars"] = cfg.symbol_cooldown_bars
    return out


def extract_long_signals(prepared: pd.DataFrame, cfg: RiskOffGateConfig = RiskOffGateConfig()) -> pd.DataFrame:
    """CIC long-entry signals from the prepared frame (bullish shock + a CIC gate)."""
    if prepared.empty:
        return pd.DataFrame(columns=["exchange", "symbol", "candidate", "signal_time"])
    shock = prepared.get("bullish_volume_shock_event", pd.Series(False, index=prepared.index)).fillna(False).astype(bool)
    rank_ok = pd.to_numeric(prepared.get("dynamic_all_rank"), errors="coerce") <= cfg.top_n
    rows = []
    for candidate, gate_col in LONG_GATE_COLUMNS.items():
        if gate_col not in prepared.columns:
            continue
        mask = shock & rank_ok & prepared[gate_col].fillna(False).astype(bool)
        sample = prepared.loc[mask, ["exchange", "symbol", "feature_time"]].copy()
        if sample.empty:
            continue
        sample["candidate"] = candidate
        sample = sample.rename(columns={"feature_time": "signal_time"})
        rows.append(sample)
    if not rows:
        return pd.DataFrame(columns=["exchange", "symbol", "candidate", "signal_time"])
    return pd.concat(rows, ignore_index=True)


def write_risk_off_shadow(
    prepared: pd.DataFrame,
    report_root: Path,
    cfg: RiskOffGateConfig = RiskOffGateConfig(),
) -> dict[str, Path]:
    """End-to-end shadow: events + annotated long signals + a one-line summary.

    Observational only — writes CSVs next to the live report, changes no trade.
    """
    report_root = ensure_dir(report_root)
    events = build_risk_off_events(prepared, cfg)
    signals = extract_long_signals(prepared, cfg)
    annotated = annotate_signals_with_risk_off(signals, events, cfg)
    outputs = {
        "risk_off_events": report_root / "risk_off_events.csv",
        "risk_off_signal_shadow": report_root / "risk_off_signal_shadow.csv",
        "risk_off_status": report_root / "risk_off_status.md",
    }
    events.to_csv(outputs["risk_off_events"], index=False)
    annotated.to_csv(outputs["risk_off_signal_shadow"], index=False)
    suppressed_n = int(annotated.get("risk_off_suppressed", pd.Series(dtype=bool)).fillna(False).sum())
    total = int(len(annotated))
    lines = [
        "# Long Risk-Off Shadow (v1.2s2 gate)",
        "",
        "- mode: shadow_only — records would-be risk-off decisions, changes no trade.",
        f"- gate motifs: {', '.join(cfg.motifs)}, symbol cooldown: {cfg.symbol_cooldown_bars} bars",
        f"- failure-motif events: {len(events)}",
        f"- long signals annotated: {total}",
        f"- would be risk-off suppressed: {suppressed_n} ({(suppressed_n / total * 100) if total else 0:.1f}%)",
        "",
        "A suppressed signal means a new long on that name would be skipped because the",
        "name had a failure motif within the cooldown. No paper-live / real-live change.",
    ]
    outputs["risk_off_status"].write_text("\n".join(lines), encoding="utf-8")
    return outputs


__all__ = [
    "LONG_GATE_COLUMNS",
    "RiskOffGateConfig",
    "annotate_signals_with_risk_off",
    "build_risk_off_events",
    "extract_long_signals",
    "risk_off_decision",
    "write_risk_off_shadow",
]
