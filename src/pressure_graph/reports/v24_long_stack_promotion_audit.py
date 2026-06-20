"""v2.4 Long Stack Promotion Audit.

This report audits the current long stack as a portfolio system. It promotes
nothing by itself; it only converts the forward ledgers into sample sufficiency,
risk-envelope, and promotion-decision tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, read_parquet


REPORT_ROOT = Path("reports/v2_4_long_stack_promotion_audit")
V23_ROOT = Path("reports/v2_3_forward_evaluation_decision_ledger")
SOURCE_ROOT = Path("reports/v0_7d2_cic_mir1_paper_live")
FAILURE_OVERLAY_ROOT = Path("reports/v3_0_symbol_risk_off_overlay")


@dataclass(frozen=True)
class StackSpec:
    structure_id: str
    portfolio_id: str
    label: str
    requires_cp: bool = False
    requires_protect: bool = False
    requires_overflow: bool = False


STACKS = (
    StackSpec("S0", "P2_MAX8_BASELINE", "P2 max8"),
    StackSpec("S1", "P2_MAX8_PLUS_O6", "P2 max8 + O6", requires_overflow=True),
    StackSpec("S2", "P2_MAX8_CP60", "P2 max8 + CP60", requires_cp=True),
    StackSpec(
        "S3",
        "P2_MAX8_CP60_PLUS_O6",
        "P2 max8 + CP60 + O6",
        requires_cp=True,
        requires_overflow=True,
    ),
    StackSpec(
        "S4",
        "P2_MAX8_CP60_PROTECT_A_CAP2",
        "P2 max8 + Protect_A cap2",
        requires_cp=True,
        requires_protect=True,
    ),
    StackSpec(
        "S5",
        "P2_MAX8_CP60_PROTECT_A_CAP2_PLUS_O6",
        "P2 max8 + Protect_A cap2 + O6",
        requires_cp=True,
        requires_protect=True,
        requires_overflow=True,
    ),
)


@dataclass(frozen=True)
class V24Config:
    report_root: Path = REPORT_ROOT
    v23_root: Path = V23_ROOT
    source_root: Path = SOURCE_ROOT
    failure_overlay_root: Path = FAILURE_OVERLAY_ROOT
    core_trade_threshold: int = 100
    cp_exit_threshold: int = 50
    protected_exit_threshold: int = 30
    overflow_trade_threshold: int = 30
    exposure_cap_units: float = 10.0


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    return read_parquet(path) if path.exists() else pd.DataFrame()


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(default, index=frame.index)), errors="coerce")


def _bool(frame: pd.DataFrame, col: str) -> pd.Series:
    values = frame.get(col, pd.Series(False, index=frame.index))
    if values.dtype == object:
        return values.astype(str).str.lower().isin(["true", "1", "yes"])
    return values.fillna(False).astype(bool)


def _dt(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(frame.get(col, pd.Series(pd.NaT, index=frame.index)), utc=True, errors="coerce")


def _sum(frame: pd.DataFrame, col: str) -> float:
    values = _num(frame, col, 0.0)
    return float(values.fillna(0.0).sum()) if len(values) else 0.0


def _max(frame: pd.DataFrame, col: str, default: float = 0.0) -> float:
    values = _num(frame, col)
    return float(values.max()) if values.notna().any() else default


def _min(frame: pd.DataFrame, col: str) -> float:
    values = _num(frame, col)
    return float(values.min()) if values.notna().any() else np.nan


def _max_drawdown(daily_net: pd.Series) -> float:
    values = pd.to_numeric(daily_net, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan
    cumulative = np.cumsum(values)
    running_max = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    drawdown = cumulative - running_max
    return float(drawdown.min()) if len(drawdown) else 0.0


def _month_cap35(monthly_net: pd.Series) -> float:
    values = pd.to_numeric(monthly_net, errors="coerce").dropna()
    if values.empty:
        return np.nan
    positive = values.clip(lower=0.0)
    positive_sum = float(positive.sum())
    if positive_sum <= 0:
        return float(values.sum())
    capped_positive = positive.clip(upper=0.35 * positive_sum).sum()
    return float(capped_positive + values.clip(upper=0.0).sum())


def _max_positive_contribution(values: pd.Series) -> float:
    positive = pd.to_numeric(values, errors="coerce").dropna().clip(lower=0.0)
    total = float(positive.sum())
    if total <= 0:
        return np.nan
    return float(positive.max() / total)


def _weighted_selected(checkpoint: pd.DataFrame, portfolio_id: str) -> pd.DataFrame:
    if checkpoint.empty or "portfolio_id" not in checkpoint.columns:
        return pd.DataFrame()
    sample = checkpoint[checkpoint["portfolio_id"].astype(str).eq(portfolio_id)].copy()
    if sample.empty:
        return sample
    if "selected" in sample.columns:
        sample = sample[_bool(sample, "selected")].copy()
    if "position_size" not in sample.columns:
        sample["position_size"] = 1.0
    sample["weighted_net20"] = _num(sample, "effective_net_return_20bp", 0.0).fillna(
        _num(sample, "net_return_20bp", 0.0)
    ) * _num(sample, "position_size", 1.0).fillna(1.0)
    return sample


def _supplement_from_checkpoint(checkpoint: pd.DataFrame, portfolio_id: str) -> dict[str, float]:
    sample = _weighted_selected(checkpoint, portfolio_id)
    if sample.empty:
        return {
            "net50": np.nan,
            "worst_burst": np.nan,
            "max_symbol_contribution": np.nan,
        }
    net50 = np.nan
    if "effective_net_return_50bp" in sample.columns:
        net50 = float((_num(sample, "effective_net_return_50bp", 0.0) * _num(sample, "position_size", 1.0)).sum() / 8.0)
    worst_burst = np.nan
    if "burst_id" in sample.columns:
        burst = sample.groupby("burst_id", dropna=False)["weighted_net20"].sum() / 8.0
        worst_burst = float(burst.min()) if not burst.empty else np.nan
    symbol = sample.groupby("symbol", dropna=False)["weighted_net20"].sum() if "symbol" in sample.columns else pd.Series(dtype=float)
    return {
        "net50": net50,
        "worst_burst": worst_burst,
        "max_symbol_contribution": _max_positive_contribution(symbol),
    }


def _stack_comparison(architecture: pd.DataFrame, checkpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = architecture.copy()
    if not data.empty and "date" in data.columns:
        data["date_ts"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    for spec in STACKS:
        sample = data[data.get("structure", pd.Series(dtype=str)).astype(str).eq(spec.portfolio_id)].copy()
        if not sample.empty:
            if "date_ts" in sample.columns:
                period_dates = sample["date_ts"].dt.tz_localize(None)
                month = sample.groupby(period_dates.dt.to_period("M"))["net20"].sum()
                week = sample.groupby(period_dates.dt.to_period("W"))["net20"].sum()
            else:
                month = pd.Series(dtype=float)
                week = pd.Series(dtype=float)
            supplement = _supplement_from_checkpoint(checkpoint, spec.portfolio_id)
            row = {
                "structure_id": spec.structure_id,
                "structure": spec.portfolio_id,
                "label": spec.label,
                "trades": int(_sum(sample, "trades")),
                "net10": _sum(sample, "net10"),
                "net20": _sum(sample, "net20"),
                "net30": _sum(sample, "net30"),
                "net50": supplement["net50"],
                "core_pnl": _sum(sample, "core_pnl"),
                "overflow_pnl": _sum(sample, "overflow_pnl"),
                "checkpoint_delta": _sum(sample, "checkpoint_pnl"),
                "protect_delta": _sum(sample, "protect_counterfactual_pnl"),
                "max_exposure": _max(sample, "max_exposure"),
                "max_concurrent_positions": _max(sample, "max_concurrent_positions"),
                "worst_burst": supplement["worst_burst"],
                "worst_day": _min(sample, "net20"),
                "worst_week": float(week.min()) if not week.empty else np.nan,
                "worst_month": float(month.min()) if not month.empty else np.nan,
                "max_drawdown_proxy": _max_drawdown(_num(sample.sort_values("date_ts"), "net20")),
                "month_cap35_net20": _month_cap35(month),
                "max_month_contribution": _max_positive_contribution(month),
                "max_symbol_contribution": supplement["max_symbol_contribution"],
                "checkpoint_exits": int(_sum(sample, "checkpoint_exits")),
                "protected_exits": int(_sum(sample, "protected_exits")),
                "overflow_trades": int(_sum(sample, "overflow_trades")),
            }
        else:
            row = {
                "structure_id": spec.structure_id,
                "structure": spec.portfolio_id,
                "label": spec.label,
                "trades": 0,
                "net10": np.nan,
                "net20": np.nan,
                "net30": np.nan,
                "net50": np.nan,
                "core_pnl": np.nan,
                "overflow_pnl": np.nan,
                "checkpoint_delta": np.nan,
                "protect_delta": np.nan,
                "max_exposure": np.nan,
                "max_concurrent_positions": np.nan,
                "worst_burst": np.nan,
                "worst_day": np.nan,
                "worst_week": np.nan,
                "worst_month": np.nan,
                "max_drawdown_proxy": np.nan,
                "month_cap35_net20": np.nan,
                "max_month_contribution": np.nan,
                "max_symbol_contribution": np.nan,
                "checkpoint_exits": 0,
                "protected_exits": 0,
                "overflow_trades": 0,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _status_label(count: int, threshold: int, applicable: bool = True) -> str:
    if not applicable:
        return "not_applicable"
    if count >= threshold:
        return "sufficient"
    return "insufficient"


def _forward_sample_sufficiency(stack: pd.DataFrame, cfg: V24Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = {spec.structure_id: spec for spec in STACKS}
    for row in stack.itertuples(index=False):
        spec = specs[str(row.structure_id)]
        core_ok = int(row.trades) >= cfg.core_trade_threshold
        cp_ok = (not spec.requires_cp) or int(row.checkpoint_exits) >= cfg.cp_exit_threshold
        protect_ok = (not spec.requires_protect) or int(row.protected_exits) >= cfg.protected_exit_threshold
        overflow_ok = (not spec.requires_overflow) or int(row.overflow_trades) >= cfg.overflow_trade_threshold
        rows.append(
            {
                "structure_id": row.structure_id,
                "structure": row.structure,
                "label": row.label,
                "core_trades": int(row.trades),
                "core_trade_threshold": cfg.core_trade_threshold,
                "core_sample_status": _status_label(int(row.trades), cfg.core_trade_threshold),
                "cp_exits": int(row.checkpoint_exits),
                "cp_exit_threshold": cfg.cp_exit_threshold if spec.requires_cp else np.nan,
                "cp_sample_status": _status_label(int(row.checkpoint_exits), cfg.cp_exit_threshold, spec.requires_cp),
                "protected_exits": int(row.protected_exits),
                "protected_exit_threshold": cfg.protected_exit_threshold if spec.requires_protect else np.nan,
                "protect_sample_status": _status_label(int(row.protected_exits), cfg.protected_exit_threshold, spec.requires_protect),
                "overflow_trades": int(row.overflow_trades),
                "overflow_trade_threshold": cfg.overflow_trade_threshold if spec.requires_overflow else np.nan,
                "overflow_sample_status": _status_label(int(row.overflow_trades), cfg.overflow_trade_threshold, spec.requires_overflow),
                "overall_sample_status": "sufficient" if core_ok and cp_ok and protect_ok and overflow_ok else "insufficient",
            }
        )
    return pd.DataFrame(rows)


def _risk_envelope_check(stack: pd.DataFrame, cfg: V24Config) -> pd.DataFrame:
    s0 = stack[stack["structure_id"].eq("S0")]
    baseline_worst_month = float(s0["worst_month"].iloc[0]) if not s0.empty else np.nan
    baseline_dd = float(s0["max_drawdown_proxy"].iloc[0]) if not s0.empty else np.nan
    rows: list[dict[str, Any]] = []
    for row in stack.itertuples(index=False):
        exposure_ok = pd.isna(row.max_exposure) or float(row.max_exposure) <= cfg.exposure_cap_units
        month_cap_ok = pd.notna(row.month_cap35_net20) and float(row.month_cap35_net20) > 0
        net30_ok = pd.notna(row.net30) and float(row.net30) > 0
        worst_month_ok = pd.isna(baseline_worst_month) or pd.isna(row.worst_month) or float(row.worst_month) >= baseline_worst_month
        dd_ok = pd.isna(baseline_dd) or pd.isna(row.max_drawdown_proxy) or float(row.max_drawdown_proxy) >= baseline_dd
        hard_pass = bool(exposure_ok and month_cap_ok and net30_ok and worst_month_ok and dd_ok)
        rows.append(
            {
                "structure_id": row.structure_id,
                "structure": row.structure,
                "label": row.label,
                "max_exposure": row.max_exposure,
                "exposure_cap": cfg.exposure_cap_units,
                "exposure_ok": exposure_ok,
                "net30": row.net30,
                "net30_positive": net30_ok,
                "month_cap35_net20": row.month_cap35_net20,
                "month_cap35_positive": month_cap_ok,
                "worst_month": row.worst_month,
                "baseline_worst_month": baseline_worst_month,
                "worst_month_not_worse_than_baseline": worst_month_ok,
                "max_drawdown_proxy": row.max_drawdown_proxy,
                "baseline_max_drawdown_proxy": baseline_dd,
                "drawdown_not_worse_than_baseline": dd_ok,
                "risk_envelope_status": "pass" if hard_pass else "not_passed",
            }
        )
    return pd.DataFrame(rows)


def _breakdown(stack: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "structure_id",
        "structure",
        "label",
        "trades",
        "net20",
        "core_pnl",
        "overflow_pnl",
        "checkpoint_delta",
        "protect_delta",
        "checkpoint_exits",
        "protected_exits",
        "overflow_trades",
    ]
    return stack.reindex(columns=cols)


def _failure_overlay_interaction(cfg: V24Config) -> pd.DataFrame:
    path = cfg.failure_overlay_root / "architecture_overlay_summary.csv"
    data = _read_csv(path)
    columns = [
        "structure_id",
        "base_structure",
        "risk_off_structure",
        "risk_off_gated_candidates",
        "risk_off_gated_net20_avg",
        "risk_off_false_skip_rate",
        "delta_net20_vs_base",
        "delta_drawdown_vs_base",
        "interaction_status",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    mapping = {
        "R0_P2_MAX8": ("S0", "B0_P2_MAX8"),
        "R1_P2_MAX8_O6": ("S1", "B1_P2_MAX8_O6"),
        "R2_P2_MAX8_CP60_O6": ("S3", "B2_P2_MAX8_CP60_O6"),
        "R3_P2_MAX8_PROTECT_A_CAP2_O6": ("S5", "B3_P2_MAX8_PROTECT_A_CAP2_O6"),
    }
    rows: list[dict[str, Any]] = []
    for risk_id, (structure_id, base_id) in mapping.items():
        risk = data[data.get("structure_id", pd.Series(dtype=str)).astype(str).eq(risk_id)]
        if risk.empty:
            continue
        r = risk.iloc[0]
        delta_net = pd.to_numeric(r.get("delta_net20_vs_base"), errors="coerce")
        delta_dd = pd.to_numeric(r.get("delta_drawdown_vs_base"), errors="coerce")
        if pd.notna(delta_net) and delta_net > 0 and pd.notna(delta_dd) and delta_dd >= 0:
            status = "improves_net_and_drawdown"
        elif pd.notna(delta_dd) and delta_dd >= 0:
            status = "diagnostic_drawdown_help_net_not_improved"
        else:
            status = "diagnostic_only"
        rows.append(
            {
                "structure_id": structure_id,
                "base_structure": base_id,
                "risk_off_structure": risk_id,
                "risk_off_gated_candidates": r.get("risk_off_gated_candidates"),
                "risk_off_gated_net20_avg": r.get("risk_off_gated_net20_avg"),
                "risk_off_false_skip_rate": r.get("risk_off_false_skip_rate"),
                "delta_net20_vs_base": delta_net,
                "delta_drawdown_vs_base": delta_dd,
                "interaction_status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _low_coimpulse_diagnostic(cfg: V24Config) -> pd.DataFrame:
    data = _read_csv(cfg.v23_root / "live_regime_diagnostics.csv")
    columns = [
        "risk_state",
        "events",
        "net20_avg",
        "net20_sum",
        "hit_rate",
        "sample_status",
        "diagnostic_status",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for state, group in data.groupby("risk_state", dropna=False):
        net = _num(group, "net20_later")
        rows.append(
            {
                "risk_state": state,
                "events": int(len(group)),
                "net20_avg": float(net.mean()) if net.notna().any() else np.nan,
                "net20_sum": float(net.fillna(0.0).sum()),
                "hit_rate": float(net.gt(0).mean()) if len(net) else np.nan,
                "sample_status": "sufficient" if len(group) >= 100 else "insufficient",
                "diagnostic_status": "diagnostic_only_no_live_action",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _decision_for(
    row: pd.Series,
    suff: pd.Series,
    risk: pd.Series,
    s3_net20: float | None,
    s5_net20: float | None,
) -> str:
    structure_id = str(row["structure_id"])
    if int(row.get("trades", 0)) == 0:
        return "NEED_MORE_SAMPLE"
    if structure_id == "S0":
        return "KEEP_REFERENCE_CORE_BASELINE"
    if str(suff.get("overall_sample_status")) != "sufficient":
        if structure_id in {"S3"}:
            return "KEEP_SHADOW_CONSERVATIVE_NEED_MORE_SAMPLE"
        if structure_id in {"S5"}:
            return "KEEP_SHADOW_RESEARCH_IMPROVED_NEED_PROTECT_OR_OVERFLOW_SAMPLE"
        return "KEEP_SHADOW_NEED_MORE_SAMPLE"
    if str(risk.get("risk_envelope_status")) != "pass":
        return "KEEP_SHADOW_RISK_ENVELOPE_NOT_PASSED"
    if structure_id == "S5" and s3_net20 is not None and s5_net20 is not None:
        return "PROMOTE_TO_SHADOW_CANDIDATE" if s5_net20 > s3_net20 else "KEEP_SHADOW_DO_NOT_REPLACE_S3"
    if structure_id == "S3":
        return "PROMOTE_TO_CONSERVATIVE_LONG_PAPER_STACK"
    return "KEEP_SHADOW"


def _promotion_decisions(stack: pd.DataFrame, sufficiency: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    s3 = stack[stack["structure_id"].eq("S3")]
    s5 = stack[stack["structure_id"].eq("S5")]
    s3_net20 = float(s3["net20"].iloc[0]) if not s3.empty and pd.notna(s3["net20"].iloc[0]) else None
    s5_net20 = float(s5["net20"].iloc[0]) if not s5.empty and pd.notna(s5["net20"].iloc[0]) else None
    rows = []
    suff_map = {str(row.structure_id): row._asdict() for row in sufficiency.itertuples(index=False)}
    risk_map = {str(row.structure_id): row._asdict() for row in risk.itertuples(index=False)}
    for _, row in stack.iterrows():
        sid = str(row["structure_id"])
        suff = pd.Series(suff_map.get(sid, {}))
        risk_row = pd.Series(risk_map.get(sid, {}))
        rows.append(
            {
                "structure_id": sid,
                "structure": row["structure"],
                "label": row["label"],
                "net20": row["net20"],
                "trades": row["trades"],
                "checkpoint_exits": row["checkpoint_exits"],
                "protected_exits": row["protected_exits"],
                "overflow_trades": row["overflow_trades"],
                "sample_status": suff.get("overall_sample_status", "insufficient"),
                "risk_envelope_status": risk_row.get("risk_envelope_status", "not_passed"),
                "decision": _decision_for(row, suff, risk_row, s3_net20, s5_net20),
            }
        )
    return pd.DataFrame(rows)


def _write_notes(root: Path, decisions: pd.DataFrame, sufficiency: pd.DataFrame) -> None:
    s3 = decisions[decisions["structure_id"].eq("S3")]
    s5 = decisions[decisions["structure_id"].eq("S5")]
    s3_decision = s3["decision"].iloc[0] if not s3.empty else "missing"
    s5_decision = s5["decision"].iloc[0] if not s5.empty else "missing"
    insufficient = sufficiency[sufficiency["overall_sample_status"].ne("sufficient")]
    lines = [
        "# v2.4 Long Stack Promotion Audit",
        "",
        "Status: audit only. No paper-live primary, shadow action, or real-live permission is changed.",
        "",
        "## Primary Question",
        "- Compare S3 (P2 max8 + CP60 + O6) against S5 (P2 max8 + Protect_A cap2 + O6).",
        "- Promote only when forward sample, risk envelope, and component contributions are all sufficient.",
        "",
        "## Current Decision",
        f"- S3 decision: {s3_decision}",
        f"- S5 decision: {s5_decision}",
        "- Real-live remains disabled.",
    ]
    if not insufficient.empty:
        lines.extend(["", "## Sample Gaps"])
        for row in insufficient.itertuples(index=False):
            lines.append(
                f"- {row.structure_id} {row.label}: core={row.core_trades}, "
                f"cp={row.cp_exits}, protected={row.protected_exits}, overflow={row.overflow_trades}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- P2 max8 remains the core baseline/reference.",
            "- O6, CP60, and Protect_A cap2 must be judged by forward component samples, not only historical lift.",
            "- Low-coimpulse and failure/risk-off overlays remain diagnostics unless a separate shadow promotion audit passes.",
        ]
    )
    root.joinpath("promotion_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v24_long_stack_promotion_audit(cfg: V24Config = V24Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    architecture = _read_csv(cfg.v23_root / "live_architecture_summary.csv")
    checkpoint = _read_parquet(cfg.source_root / "checkpoint_trade_ledger.parquet")
    stack = _stack_comparison(architecture, checkpoint)
    sufficiency = _forward_sample_sufficiency(stack, cfg)
    breakdown = _breakdown(stack)
    risk = _risk_envelope_check(stack, cfg)
    failure = _failure_overlay_interaction(cfg)
    low_coimpulse = _low_coimpulse_diagnostic(cfg)
    decisions = _promotion_decisions(stack, sufficiency, risk)

    outputs = {
        "stack_comparison": root / "stack_comparison.csv",
        "forward_sample_sufficiency": root / "forward_sample_sufficiency.csv",
        "core_overflow_checkpoint_breakdown": root / "core_overflow_checkpoint_breakdown.csv",
        "risk_envelope_check": root / "risk_envelope_check.csv",
        "failure_overlay_interaction": root / "failure_overlay_interaction.csv",
        "low_coimpulse_diagnostic": root / "low_coimpulse_diagnostic.csv",
        "promotion_decision_table": root / "promotion_decision_table.csv",
        "promotion_decision": root / "promotion_decision.md",
    }
    stack.to_csv(outputs["stack_comparison"], index=False)
    sufficiency.to_csv(outputs["forward_sample_sufficiency"], index=False)
    breakdown.to_csv(outputs["core_overflow_checkpoint_breakdown"], index=False)
    risk.to_csv(outputs["risk_envelope_check"], index=False)
    failure.to_csv(outputs["failure_overlay_interaction"], index=False)
    low_coimpulse.to_csv(outputs["low_coimpulse_diagnostic"], index=False)
    decisions.to_csv(outputs["promotion_decision_table"], index=False)
    _write_notes(root, decisions, sufficiency)
    return outputs


__all__ = ["V24Config", "write_v24_long_stack_promotion_audit"]
