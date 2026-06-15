"""v3.0 Symbol Risk-Off Overlay.

This report integrates the short/failure research as a long-book risk overlay.
It does not open shorts and does not change paper-live permissions. The first
question is narrower: if a symbol prints a confirmed failure motif, should the
current long stack stop opening new longs on that same symbol for 48 bars?
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig
from pressure_graph.reports.v12s2_long_risk_off_overlay import (
    BAR_NS,
    RiskOffConfig,
    _apply_symbol_gate,
    _epoch_ns,
    stream_risk_off_events,
)
from pressure_graph.reports.v13a_checkpoint_robustness import CORE_MAX_POSITIONS
from pressure_graph.reports.v13d_cp60_context_protection import (
    _apply_protected_checkpoint,
    _portfolio_summary,
    _simulate_core_max8,
)
from pressure_graph.reports.v13e_cp60_beta_protection_stability import (
    _cp60_sample,
    _portfolio_net,
    _prepare_sample_at_cost,
    _simulate_max8_o6,
)
from pressure_graph.reports.v13f_cp60_protect_a_stability import _cap_mask_by_burst


REPORT_ROOT = Path("reports/v3_0_symbol_risk_off_overlay")
FOCAL_COST_BPS = 20.0


@dataclass(frozen=True)
class V30Config:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    top_n: int = 30
    motifs: tuple[str, ...] = ("S1", "S3", "S5")
    symbol_cooldown_bars: int = 48


def _risk_cfg(cfg: V30Config) -> RiskOffConfig:
    return RiskOffConfig(
        report_root=cfg.report_root,
        top_n=cfg.top_n,
        motifs=cfg.motifs,
        symbol_cooldown_bars=cfg.symbol_cooldown_bars,
    )


def _prepare_hold_sample(sample: pd.DataFrame) -> pd.DataFrame:
    """Use original vol_regime_fast exits with no CP60 management."""
    out = sample.copy()
    out["protection_rule"] = "none"
    out["would_have_exited_at_cp60"] = False
    out["kept_due_to_protection"] = False
    out["checkpoint_early_exit"] = False
    out["effective_exit_time"] = out["exit_time"]
    out["effective_net_return"] = out["net_return_at_cost"]
    out["effective_holding_minutes"] = (
        pd.to_datetime(out["effective_exit_time"], utc=True, errors="coerce")
        - pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    return out


def _prepare_protect_a_cap2_sample(sample: pd.DataFrame) -> pd.DataFrame:
    mask = _cap_mask_by_burst(sample, 2)
    return _apply_protected_checkpoint(sample, mask, "Protect_A_beta_high_cap2")


def _drawdown(contribution: pd.Series) -> float:
    if contribution.empty:
        return np.nan
    equity = contribution.cumsum()
    return float((equity - equity.cummax()).min())


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _o6_summary(rule_name: str, ledger: pd.DataFrame, skipped: pd.DataFrame) -> dict[str, object]:
    weighted = _num(ledger, "weighted_return", 0.0)
    contribution = weighted / CORE_MAX_POSITIONS
    overflow = (
        ledger.get("sleeve", pd.Series(dtype=str)).astype(str).eq("overflow")
        if not ledger.empty
        else pd.Series(dtype=bool)
    )
    protected = (
        ledger.get("kept_due_to_protection", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
        if not ledger.empty
        else pd.Series(dtype=bool)
    )
    raw_cp = (
        ledger.get("would_have_exited_at_cp60", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
        if not ledger.empty
        else pd.Series(dtype=bool)
    )
    if not ledger.empty:
        month_key = pd.to_datetime(ledger["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
        month_return = weighted.groupby(month_key, sort=False, dropna=False).sum() / CORE_MAX_POSITIONS
        burst_key = ledger.get("burst_id", pd.Series("unknown", index=ledger.index)).astype(str)
        burst_return = weighted.groupby(burst_key, sort=False, dropna=False).sum() / CORE_MAX_POSITIONS
    else:
        month_return = pd.Series(dtype=float)
        burst_return = pd.Series(dtype=float)
    return {
        "rule": rule_name,
        "selected_trades": int(len(ledger)),
        "skipped_trades": int(len(skipped)),
        "portfolio_net20": _portfolio_net(ledger),
        "selected_effective_net20": float(_num(ledger, "effective_net_return").mean()) if len(ledger) else np.nan,
        "skipped_counterfactual_net20": float(_num(skipped, "net_return_at_cost").mean()) if len(skipped) else np.nan,
        "selected_minus_skipped": (
            float(_num(ledger, "effective_net_return").mean() - _num(skipped, "net_return_at_cost").mean())
            if len(ledger) and len(skipped)
            else np.nan
        ),
        "month_cap35_net20": np.nan,
        "worst_month": float(month_return.min()) if not month_return.empty else np.nan,
        "worst_burst": float(burst_return.min()) if not burst_return.empty else np.nan,
        "max_drawdown_proxy": _drawdown(contribution),
        "capital_utilization": np.nan,
        "cp60_exits_executed": int((raw_cp & ~protected).sum()) if len(raw_cp) else 0,
        "protected_cp60_exits": int(protected.sum()) if len(protected) else 0,
        "overflow_trades": int(overflow.sum()) if len(overflow) else 0,
        "overflow_weighted_net20": float(_num(ledger[overflow], "weighted_return", 0.0).sum() / CORE_MAX_POSITIONS)
        if len(overflow)
        else 0.0,
        "max_exposure_weight": float(_num(ledger, "exposure_weight", 1.0).max()) if len(ledger) else 0.0,
    }


def _simulate_architecture(sample: pd.DataFrame, structure_id: str, *, use_o6: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if use_o6:
        ledger, skipped = _simulate_max8_o6(sample, structure_id)
        return ledger, skipped, _o6_summary(structure_id, ledger, skipped)
    ledger, skipped = _simulate_core_max8(sample, structure_id)
    summary = _portfolio_summary(structure_id, ledger, skipped)
    summary["overflow_trades"] = 0
    summary["overflow_weighted_net20"] = 0.0
    summary["max_exposure_weight"] = 1.0 if not ledger.empty else 0.0
    return ledger, skipped, summary


def _architecture_samples(base_sample: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, bool, str]]:
    hold = _prepare_hold_sample(base_sample)
    cp60 = _cp60_sample(base_sample)
    protect = _prepare_protect_a_cap2_sample(base_sample)
    return {
        "B0_P2_MAX8": (hold, False, "P2 max8"),
        "B1_P2_MAX8_O6": (hold, True, "P2 max8 + O6"),
        "B2_P2_MAX8_CP60_O6": (cp60, True, "P2 max8 + CP60 + O6"),
        "B3_P2_MAX8_PROTECT_A_CAP2_O6": (protect, True, "P2 max8 + Protect_A cap2 + O6"),
    }


def _risk_event_details(pool: pd.DataFrame, events: pd.DataFrame, cfg: RiskOffConfig) -> pd.DataFrame:
    columns = [
        "risk_off_active",
        "risk_off_event_time",
        "risk_off_event_age_bars",
        "risk_off_motifs",
        "risk_off_event_count",
    ]
    out = pd.DataFrame(index=pool.index, columns=columns)
    out["risk_off_active"] = False
    out["risk_off_event_time"] = pd.Series(pd.NaT, index=pool.index, dtype="datetime64[ns, UTC]")
    out["risk_off_event_age_bars"] = np.nan
    out["risk_off_motifs"] = ""
    out["risk_off_event_count"] = 0
    if events.empty or pool.empty:
        return out

    window_ns = cfg.symbol_cooldown_bars * BAR_NS
    events = events.copy()
    events["feature_time"] = pd.to_datetime(events["feature_time"], utc=True, errors="coerce")
    events = events.dropna(subset=["feature_time"]).sort_values("feature_time")
    events_ns = _epoch_ns(events["feature_time"])
    by_symbol = {
        str(sym): idx.to_numpy()
        for sym, idx in events.reset_index(drop=True).groupby("symbol").groups.items()
    }
    pool_ns = _epoch_ns(pool["signal_time"])
    symbols = pool["symbol"].astype(str).to_numpy()
    for pos, idx in enumerate(pool.index):
        event_idx = by_symbol.get(symbols[pos])
        if event_idx is None:
            continue
        local_ns = events_ns[event_idx]
        signal = pool_ns[pos]
        left = np.searchsorted(local_ns, signal - window_ns, side="right")
        right = np.searchsorted(local_ns, signal, side="right")
        if right <= left:
            continue
        matched_idx = event_idx[left:right]
        matched = events.iloc[matched_idx]
        latest = matched.iloc[-1]
        age = (signal - int(pd.Timestamp(latest["feature_time"]).value)) / BAR_NS
        out.loc[idx, "risk_off_active"] = True
        out.loc[idx, "risk_off_event_time"] = latest["feature_time"]
        out.loc[idx, "risk_off_event_age_bars"] = age
        out.loc[idx, "risk_off_motifs"] = ",".join(sorted(set(matched["motif"].astype(str))))
        out.loc[idx, "risk_off_event_count"] = int(len(matched))
    return out


def _make_skipped_attribution(
    sample: pd.DataFrame,
    risk_mask: pd.Series,
    details: pd.DataFrame,
    structure_id: str,
    structure_label: str,
) -> pd.DataFrame:
    gated = sample[risk_mask.reindex(sample.index).fillna(False).to_numpy()].copy()
    if gated.empty:
        return pd.DataFrame()
    local_details = details.reindex(gated.index)
    out = gated.copy()
    out["structure_id"] = structure_id
    out["structure_label"] = structure_label
    out["would_be_net20"] = _num(out, "effective_net_return")
    out["would_be_raw_net20"] = _num(out, "net_return_at_cost")
    out["risk_off_false_skip"] = out["would_be_net20"].gt(0.0)
    for col in local_details.columns:
        out[col] = local_details[col].to_numpy()
    preferred = [
        "structure_id",
        "structure_label",
        "signal_id",
        "trade_key",
        "symbol",
        "candidate",
        "signal_time",
        "entry_time",
        "would_be_net20",
        "would_be_raw_net20",
        "risk_off_false_skip",
        "risk_off_event_time",
        "risk_off_event_age_bars",
        "risk_off_motifs",
        "risk_off_event_count",
    ]
    cols = [col for col in preferred if col in out.columns]
    return out[cols + [col for col in out.columns if col not in cols]]


def _summary_row(
    structure_id: str,
    structure_label: str,
    variant: str,
    summary: dict[str, object],
    gated_sample: pd.DataFrame,
    risk_details: pd.DataFrame,
) -> dict[str, object]:
    gated_net = _num(gated_sample, "effective_net_return")
    row = {
        "structure_id": structure_id,
        "structure_label": structure_label,
        "variant": variant,
        "risk_off_overlay": variant == "symbol_risk_off_48",
        "selected_trades": int(summary.get("selected_trades", 0)),
        "skipped_trades": int(summary.get("skipped_trades", 0)),
        "portfolio_net20": float(summary.get("portfolio_net20", np.nan)),
        "selected_effective_net20": float(summary.get("selected_effective_net20", np.nan)),
        "skipped_counterfactual_net20": float(summary.get("skipped_counterfactual_net20", np.nan)),
        "selected_minus_skipped": float(summary.get("selected_minus_skipped", np.nan)),
        "month_cap35_net20": float(summary.get("month_cap35_net20", np.nan)),
        "worst_month": float(summary.get("worst_month", np.nan)),
        "worst_burst": float(summary.get("worst_burst", np.nan)),
        "max_drawdown_proxy": float(summary.get("max_drawdown_proxy", np.nan)),
        "capital_utilization": float(summary.get("capital_utilization", np.nan)),
        "cp60_exits_executed": int(summary.get("cp60_exits_executed", 0)),
        "protected_cp60_exits": int(summary.get("protected_cp60_exits", 0)),
        "overflow_trades": int(summary.get("overflow_trades", 0)),
        "overflow_weighted_net20": float(summary.get("overflow_weighted_net20", 0.0)),
        "risk_off_gated_candidates": int(len(gated_sample)),
        "risk_off_gated_net20_avg": float(gated_net.mean()) if len(gated_net) else np.nan,
        "risk_off_gated_loss_share": float(gated_net.lt(0.0).mean()) if len(gated_net) else np.nan,
        "risk_off_false_skip_count": int(gated_net.gt(0.0).sum()) if len(gated_net) else 0,
        "risk_off_false_skip_rate": float(gated_net.gt(0.0).mean()) if len(gated_net) else np.nan,
    }
    if not risk_details.empty and not gated_sample.empty:
        motifs = risk_details.reindex(gated_sample.index)["risk_off_motifs"].astype(str)
        row["risk_off_motif_mix"] = ";".join(
            f"{motif}:{count}" for motif, count in motifs.value_counts().sort_index().items()
        )
    else:
        row["risk_off_motif_mix"] = ""
    return row


def _overlay_summaries(
    base_sample: pd.DataFrame,
    events: pd.DataFrame,
    cfg: V30Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_cfg = _risk_cfg(cfg)
    base_mask = _apply_symbol_gate(base_sample, events, risk_cfg).reindex(base_sample.index).fillna(False).astype(bool)
    details = _risk_event_details(base_sample, events, risk_cfg)
    rows: list[dict[str, object]] = []
    skipped_frames: list[pd.DataFrame] = []
    for structure_id, (sample, use_o6, label) in _architecture_samples(base_sample).items():
        risk_mask = base_mask.reindex(sample.index).fillna(False).astype(bool)
        gated_sample = sample[risk_mask].copy()
        ledger, skipped, summary = _simulate_architecture(sample, structure_id, use_o6=use_o6)
        rows.append(_summary_row(structure_id, label, "baseline", summary, sample.iloc[0:0], details))

        overlay_id = f"R{structure_id[1:]}" if structure_id.startswith("B") else f"R_{structure_id}"
        overlay_sample = sample[~risk_mask].copy()
        overlay_ledger, overlay_skipped, overlay_summary = _simulate_architecture(overlay_sample, overlay_id, use_o6=use_o6)
        rows.append(_summary_row(overlay_id, f"{label} + symbol_risk_off_48", "symbol_risk_off_48", overlay_summary, gated_sample, details))
        skipped_attr = _make_skipped_attribution(gated_sample, pd.Series(True, index=gated_sample.index), details, overlay_id, label)
        if not skipped_attr.empty:
            skipped_frames.append(skipped_attr)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        baseline_rows = summary[summary["variant"].eq("baseline")].copy()
        baseline_rows["overlay_structure_id"] = baseline_rows["structure_id"].str.replace("^B", "R", regex=True)
        base_lookup = baseline_rows.set_index("overlay_structure_id")
        for idx, row in summary.iterrows():
            if row["variant"] != "symbol_risk_off_48":
                continue
            base = base_lookup.loc[row["structure_id"]] if row["structure_id"] in base_lookup.index else None
            if base is None:
                continue
            summary.loc[idx, "delta_net20_vs_base"] = row["portfolio_net20"] - base["portfolio_net20"]
            summary.loc[idx, "delta_drawdown_vs_base"] = row["max_drawdown_proxy"] - base["max_drawdown_proxy"]
            summary.loc[idx, "selected_trade_delta_vs_base"] = row["selected_trades"] - base["selected_trades"]
            summary.loc[idx, "overflow_trade_delta_vs_base"] = row["overflow_trades"] - base["overflow_trades"]
    skipped_attr = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    return summary, skipped_attr


def _write_notes(root: Path, summary: pd.DataFrame, events: pd.DataFrame) -> None:
    lines = [
        "# v3.0 Symbol Risk-Off Overlay",
        "",
        "Purpose: integrate short/failure motifs as a same-symbol no-long overlay on the current long stack.",
        "Status: offline integration report only. No primary, shadow, paper-live, or real-live permission changes.",
        "",
        "## Event Stream",
        f"- failure events: {len(events)}",
    ]
    if not events.empty:
        lines.append(f"- symbols with failure events: {events['symbol'].nunique()}")
        lines.append(f"- motifs: {', '.join(sorted(events['motif'].astype(str).unique()))}")
    lines.extend(["", "## Architecture Deltas"])
    overlays = summary[summary["variant"].eq("symbol_risk_off_48")].copy() if not summary.empty else pd.DataFrame()
    if overlays.empty:
        lines.append("- No overlay rows.")
    else:
        for row in overlays.itertuples(index=False):
            lines.append(
                f"- {row.structure_id}: net20={row.portfolio_net20:.4%}, "
                f"delta={getattr(row, 'delta_net20_vs_base', np.nan):+.4%}, "
                f"dd_delta={getattr(row, 'delta_drawdown_vs_base', np.nan):+.4%}, "
                f"gated={row.risk_off_gated_candidates}, "
                f"gated_avg={row.risk_off_gated_net20_avg:.4%}."
            )
    lines.extend(["", "## Interpretation"])
    if not overlays.empty:
        raw_like = overlays[overlays["structure_id"].isin(["R0_P2_MAX8", "R1_P2_MAX8_O6"])]
        managed = overlays[overlays["structure_id"].isin(["R2_P2_MAX8_CP60_O6", "R3_P2_MAX8_PROTECT_A_CAP2_O6"])]
        raw_pass = bool((raw_like["delta_net20_vs_base"] > 0).all()) if not raw_like.empty else False
        managed_net_pass = bool((managed["delta_net20_vs_base"] > 0).all()) if not managed.empty else False
        managed_dd_pass = bool((managed["delta_drawdown_vs_base"] > 0).all()) if not managed.empty else False
        lines.append(
            f"- raw P2/O6 net improvement: {'yes' if raw_pass else 'no'}; "
            f"managed-stack net improvement: {'yes' if managed_net_pass else 'no'}; "
            f"managed-stack drawdown improvement: {'yes' if managed_dd_pass else 'no'}."
        )
        if raw_pass and managed_dd_pass and not managed_net_pass:
            lines.append(
                "- Current verdict: symbol risk-off is useful risk information, but it is not yet a "
                "drop-in overlay for the CP60/Protect_A best stack because it blocks too many "
                "managed trades with positive counterfactual returns."
            )
            lines.append(
                "- Next best use: v3.1 failure-aware position management, where failure motifs act on "
                "existing weak longs / overflow / checkpoint behavior instead of blindly suppressing "
                "all future same-symbol CIC entries."
            )
    lines.extend(
        [
            "",
            "## Discipline",
            "- Gate is strict as-of: only failure confirmations with feature_time <= the long signal_time can block a long.",
            "- The overlay action is full no-long skip, not size-down and not short entry.",
            "- Market-wide risk-off is intentionally excluded from v3.0; this pass is symbol-level only.",
            "- Promotion target, if validated later, is shadow no-long gate first; real-live remains disabled.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v30_symbol_risk_off_overlay(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V30Config = V30Config(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample_at_cost(feature_path, instruments, config, root, cfg.v10a, FOCAL_COST_BPS)
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    events = stream_risk_off_events(feature_path, rank30, rank90, symbols, config, _risk_cfg(cfg))
    summary, skipped_attr = _overlay_summaries(sample, events, cfg)
    outputs = {
        "risk_off_events": root / "risk_off_events.csv",
        "architecture_overlay_summary": root / "architecture_overlay_summary.csv",
        "risk_off_skipped_attribution": root / "risk_off_skipped_attribution.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    events.to_csv(outputs["risk_off_events"], index=False)
    summary.to_csv(outputs["architecture_overlay_summary"], index=False)
    skipped_attr.to_csv(outputs["risk_off_skipped_attribution"], index=False)
    _write_notes(root, summary, events)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V30Config",
    "write_v30_symbol_risk_off_overlay",
]
