"""v2.5 Execution Realism Audit.

This audit checks whether the current long stack is ready for more realistic
execution assumptions. It does not alter any paper-live or real-live setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, read_parquet


REPORT_ROOT = Path("reports/v2_5_execution_realism_audit")
V24_ROOT = Path("reports/v2_4_long_stack_promotion_audit")
SOURCE_ROOT = Path("reports/v0_7d2_cic_mir1_paper_live")


@dataclass(frozen=True)
class V25Config:
    report_root: Path = REPORT_ROOT
    v24_root: Path = V24_ROOT
    source_root: Path = SOURCE_ROOT
    min_core_trades_for_realism: int = 100
    min_cp_exits_for_realism: int = 50
    min_overflow_trades_for_realism: int = 30


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


def _estimate_net50(row: pd.Series) -> float:
    if pd.notna(row.get("net50", np.nan)):
        return float(row["net50"])
    net10 = pd.to_numeric(row.get("net10"), errors="coerce")
    net20 = pd.to_numeric(row.get("net20"), errors="coerce")
    net30 = pd.to_numeric(row.get("net30"), errors="coerce")
    if pd.notna(net20) and pd.notna(net30):
        return float(net30 + 2.0 * (net30 - net20))
    if pd.notna(net10) and pd.notna(net20):
        return float(net20 + 3.0 * (net20 - net10))
    return np.nan


def _cost_stress(stack: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "structure_id",
        "structure",
        "label",
        "trades",
        "net10",
        "net20",
        "net30",
        "net50_est",
        "net30_positive",
        "net50_not_crash",
        "cost_stress_status",
    ]
    if stack.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, row in stack.iterrows():
        net50 = _estimate_net50(row)
        net30_positive = pd.notna(row.get("net30")) and float(row["net30"]) > 0
        net50_not_crash = pd.notna(net50) and net50 > -0.05
        if net30_positive and net50_not_crash:
            status = "pass"
        elif pd.notna(row.get("net30")):
            status = "not_passed"
        else:
            status = "insufficient_cost_columns"
        rows.append(
            {
                "structure_id": row.get("structure_id"),
                "structure": row.get("structure"),
                "label": row.get("label"),
                "trades": row.get("trades", 0),
                "net10": row.get("net10"),
                "net20": row.get("net20"),
                "net30": row.get("net30"),
                "net50_est": net50,
                "net30_positive": bool(net30_positive),
                "net50_not_crash": bool(net50_not_crash),
                "cost_stress_status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _execution_ledger_quality(checkpoint: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "portfolio_id",
        "rows",
        "selected_rows",
        "missing_entry_time",
        "missing_exit_time",
        "missing_effective_net20",
        "checkpoint_rows",
        "checkpoint_missing_price",
        "ledger_quality_status",
    ]
    if checkpoint.empty or "portfolio_id" not in checkpoint.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for portfolio_id, group in checkpoint.groupby("portfolio_id", dropna=False):
        selected = group[_bool(group, "selected")] if "selected" in group.columns else group
        cp_rows = selected[_bool(selected, "checkpoint_triggered")] if "checkpoint_triggered" in selected.columns else selected.iloc[0:0]
        missing_entry = int(pd.to_datetime(selected.get("entry_time"), utc=True, errors="coerce").isna().sum()) if "entry_time" in selected else len(selected)
        missing_exit = int(pd.to_datetime(selected.get("exit_time"), utc=True, errors="coerce").isna().sum()) if "exit_time" in selected else len(selected)
        missing_net = int(_num(selected, "effective_net_return_20bp").isna().sum()) if len(selected) else 0
        missing_cp_price = int(_num(cp_rows, "checkpoint_price").isna().sum()) if "checkpoint_price" in cp_rows.columns else int(len(cp_rows))
        status = "pass" if missing_entry == 0 and missing_exit == 0 and missing_net == 0 else "not_passed"
        rows.append(
            {
                "portfolio_id": portfolio_id,
                "rows": int(len(group)),
                "selected_rows": int(len(selected)),
                "missing_entry_time": missing_entry,
                "missing_exit_time": missing_exit,
                "missing_effective_net20": missing_net,
                "checkpoint_rows": int(len(cp_rows)),
                "checkpoint_missing_price": missing_cp_price,
                "ledger_quality_status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _execution_assumption_check(stack: pd.DataFrame, checkpoint: pd.DataFrame) -> pd.DataFrame:
    cp_exits = int(_num(stack, "checkpoint_exits", 0.0).fillna(0.0).max()) if not stack.empty else 0
    overflow = int(_num(stack, "overflow_trades", 0.0).fillna(0.0).max()) if not stack.empty else 0
    protected = int(_num(stack, "protected_exits", 0.0).fillna(0.0).max()) if not stack.empty else 0
    exact_cp_price = False
    if not checkpoint.empty and "checkpoint_price" in checkpoint.columns:
        cp = checkpoint[_bool(checkpoint, "checkpoint_triggered")]
        exact_cp_price = bool(len(cp) and _num(cp, "checkpoint_price").notna().all())
    rows = [
        {
            "check_id": "entry_price_model",
            "description": "Paper entry uses frozen next-bar / reclaim execution ledger.",
            "evidence": "entry_time and effective_net_return columns are present in ledger when source rows exist",
            "status": "audit_only",
            "blocking_for_real_live": False,
        },
        {
            "check_id": "cost_ladder",
            "description": "10/20/30bp costs are reported; 50bp is estimated when exact net50 is unavailable.",
            "evidence": "execution_cost_stress.csv",
            "status": "partial",
            "blocking_for_real_live": True,
        },
        {
            "check_id": "cp60_execution",
            "description": "CP60 exits should use checkpoint-time visible price and next executable bar/tick.",
            "evidence": f"cp_exits={cp_exits}, checkpoint_price_complete={exact_cp_price}",
            "status": "needs_more_sample" if cp_exits < 50 else "sample_ready",
            "blocking_for_real_live": cp_exits < 50,
        },
        {
            "check_id": "overflow_execution",
            "description": "O6 overflow adds exposure and must be tested under real concurrent execution.",
            "evidence": f"overflow_trades={overflow}",
            "status": "needs_more_sample" if overflow < 30 else "sample_ready",
            "blocking_for_real_live": overflow < 30,
        },
        {
            "check_id": "protect_a_execution",
            "description": "Protect_A cap2 requires protected-exit forward evidence before promotion.",
            "evidence": f"protected_exits={protected}",
            "status": "needs_more_sample" if protected < 30 else "sample_ready",
            "blocking_for_real_live": protected < 30,
        },
        {
            "check_id": "funding_and_contract_filters",
            "description": "Funding, min-notional, quantity precision, and contract filters are not finalized as execution adjustments.",
            "evidence": "not modeled in current promotion stack",
            "status": "not_ready",
            "blocking_for_real_live": True,
        },
        {
            "check_id": "depth_slippage",
            "description": "Static orderbook ranking failed; real execution still needs native depth/slippage measurement.",
            "evidence": "v0.9E static orderbook retained as diagnostic only",
            "status": "not_ready",
            "blocking_for_real_live": True,
        },
    ]
    return pd.DataFrame(rows)


def _component_turnover(stack: pd.DataFrame) -> pd.DataFrame:
    if stack.empty:
        return pd.DataFrame(
            columns=[
                "structure_id",
                "trades",
                "checkpoint_exits",
                "protected_exits",
                "overflow_trades",
                "checkpoint_exit_rate",
                "overflow_trade_rate",
                "protected_exit_rate",
            ]
        )
    out = stack[
        ["structure_id", "structure", "label", "trades", "checkpoint_exits", "protected_exits", "overflow_trades"]
    ].copy()
    trades = _num(out, "trades", 0.0).replace(0.0, np.nan)
    out["checkpoint_exit_rate"] = _num(out, "checkpoint_exits", 0.0) / trades
    out["overflow_trade_rate"] = _num(out, "overflow_trades", 0.0) / trades
    out["protected_exit_rate"] = _num(out, "protected_exits", 0.0) / trades
    return out


def _notes_text(cost: pd.DataFrame, assumptions: pd.DataFrame) -> str:
    blockers = assumptions[assumptions["blocking_for_real_live"].astype(bool)] if not assumptions.empty else pd.DataFrame()
    passed_cost = cost["cost_stress_status"].eq("pass").any() if not cost.empty else False
    lines = [
        "# v2.5 Execution Realism Audit",
        "",
        "Status: audit only. No live permission is changed.",
        "",
        f"- any_structure_passes_cost_stress: {str(bool(passed_cost)).lower()}",
        f"- real_live_blockers: {len(blockers)}",
    ]
    if not blockers.empty:
        lines.extend(["", "## Blockers"])
        for row in blockers.itertuples(index=False):
            lines.append(f"- {row.check_id}: {row.status} ({row.evidence})")
    lines.extend(
        [
            "",
            "## Decision",
            "- Current long stack is not execution-realism ready for real-live.",
            "- Continue paper/shadow logging until component sample thresholds and native execution checks are met.",
        ]
    )
    return "\n".join(lines) + "\n"


def _notes(root: Path, cost: pd.DataFrame, assumptions: pd.DataFrame) -> None:
    text = _notes_text(cost, assumptions)
    root.joinpath("execution_realism_decision.md").write_text(text, encoding="utf-8")
    root.joinpath("candidate_notes.md").write_text(text, encoding="utf-8")


def write_v25_execution_realism_audit(cfg: V25Config = V25Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    stack = _read_csv(cfg.v24_root / "stack_comparison.csv")
    checkpoint = _read_parquet(cfg.source_root / "checkpoint_trade_ledger.parquet")
    cost = _cost_stress(stack)
    ledger = _execution_ledger_quality(checkpoint)
    assumptions = _execution_assumption_check(stack, checkpoint)
    turnover = _component_turnover(stack)
    outputs = {
        "execution_cost_stress": root / "execution_cost_stress.csv",
        "execution_assumption_check": root / "execution_assumption_check.csv",
        "execution_ledger_quality": root / "execution_ledger_quality.csv",
        "component_turnover": root / "component_turnover.csv",
        "execution_realism_decision": root / "execution_realism_decision.md",
        "candidate_notes": root / "candidate_notes.md",
    }
    cost.to_csv(outputs["execution_cost_stress"], index=False)
    assumptions.to_csv(outputs["execution_assumption_check"], index=False)
    ledger.to_csv(outputs["execution_ledger_quality"], index=False)
    turnover.to_csv(outputs["component_turnover"], index=False)
    _notes(root, cost, assumptions)
    return outputs


__all__ = ["V25Config", "write_v25_execution_realism_audit"]
