"""v2.6 Risk Envelope Finalization.

This module writes the proposed risk envelope and stop conditions for the long
stack. It is a readiness artifact, not a live permission switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v2_6_risk_envelope_finalization")
V24_ROOT = Path("reports/v2_4_long_stack_promotion_audit")
V25_ROOT = Path("reports/v2_5_execution_realism_audit")


@dataclass(frozen=True)
class V26Config:
    report_root: Path = REPORT_ROOT
    v24_root: Path = V24_ROOT
    v25_root: Path = V25_ROOT
    core_max_positions: int = 8
    overflow_max_slots: int = 4
    overflow_total_exposure_cap: float = 2.0
    total_exposure_cap: float = 10.0
    daily_new_exposure_cap: float = 8.0
    rolling_4h_new_exposure_cap: float = 6.0
    recent_trade_window: int = 100
    component_window: int = 30


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(default, index=frame.index)), errors="coerce")


def _risk_policy_spec(cfg: V26Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "policy_key": "core_max_positions",
            "value": cfg.core_max_positions,
            "unit": "positions",
            "status": "draft_fixed_from_current_stack",
            "notes": "P2 CIC1+CIC2 combined basket.",
        },
        {
            "policy_key": "overflow_rule",
            "value": "O6 late-burst: portfolio_full and burst_count_so_far >= 9",
            "unit": "rule",
            "status": "shadow_only",
            "notes": "Additive sleeve; not primary replacement.",
        },
        {
            "policy_key": "overflow_max_slots",
            "value": cfg.overflow_max_slots,
            "unit": "slots",
            "status": "draft",
            "notes": "O6 cap.",
        },
        {
            "policy_key": "overflow_total_exposure_cap",
            "value": cfg.overflow_total_exposure_cap,
            "unit": "core-position units",
            "status": "draft",
            "notes": "CIC1 size 0.50, CIC2 size 0.25; cap keeps additive sleeve small.",
        },
        {
            "policy_key": "total_exposure_cap",
            "value": cfg.total_exposure_cap,
            "unit": "core-position units",
            "status": "draft",
            "notes": "Core 8 plus up to 2 overflow exposure.",
        },
        {
            "policy_key": "single_symbol_max_position",
            "value": "1 active long",
            "unit": "symbol",
            "status": "draft",
            "notes": "No same-symbol pyramiding unless separately qualified and audited.",
        },
        {
            "policy_key": "daily_new_exposure_cap",
            "value": cfg.daily_new_exposure_cap,
            "unit": "core-position units",
            "status": "proposed",
            "notes": "Needs forward validation before enforcement.",
        },
        {
            "policy_key": "rolling_4h_new_exposure_cap",
            "value": cfg.rolling_4h_new_exposure_cap,
            "unit": "core-position units",
            "status": "proposed",
            "notes": "Controls burst concentration.",
        },
        {
            "policy_key": "data_stale_stop",
            "value": "true",
            "unit": "global stop",
            "status": "required",
            "notes": "Pause all new paper/live actions when market or feature data is stale.",
        },
    ]
    return pd.DataFrame(rows)


def _stop_conditions(cfg: V26Config) -> pd.DataFrame:
    rows = [
        {
            "stop_id": "DATA_STALE_STOP",
            "scope": "global",
            "condition": "latest feature / BTC / market data is stale",
            "action": "pause all new long entries and overflow",
            "status": "required",
        },
        {
            "stop_id": "RECENT_CORE_NET_STOP",
            "scope": "core",
            "condition": f"last {cfg.recent_trade_window} core trades net20 < 0",
            "action": "freeze promotion; keep paper logging only",
            "status": "draft_threshold",
        },
        {
            "stop_id": "OVERFLOW_DEGRADATION_STOP",
            "scope": "overflow",
            "condition": f"last {cfg.component_window} overflow trades contribution < 0",
            "action": "disable O6 shadow promotion consideration",
            "status": "draft_threshold",
        },
        {
            "stop_id": "CP60_DEGRADATION_STOP",
            "scope": "checkpoint",
            "condition": f"last {cfg.component_window} CP60 exits delta <= 0",
            "action": "demote CP60 to diagnostic until revalidated",
            "status": "draft_threshold",
        },
        {
            "stop_id": "PROTECT_A_DEGRADATION_STOP",
            "scope": "protect",
            "condition": f"last {cfg.component_window} protected exits delta <= 0",
            "action": "keep CP60_all; do not upgrade Protect_A",
            "status": "draft_threshold",
        },
        {
            "stop_id": "WORST_BURST_STOP",
            "scope": "portfolio",
            "condition": "single burst loss breaches audited historical stress envelope",
            "action": "pause new same-burst entries and review",
            "status": "needs_threshold_from_v2_5",
        },
        {
            "stop_id": "COST_DRIFT_STOP",
            "scope": "execution",
            "condition": "realized all-in cost exceeds 30bp model or 50bp stress fails badly",
            "action": "disable real-live readiness; keep paper-only",
            "status": "required",
        },
    ]
    return pd.DataFrame(rows)


def _current_envelope_check(cfg: V26Config) -> pd.DataFrame:
    stack = _read_csv(cfg.v24_root / "stack_comparison.csv")
    suff = _read_csv(cfg.v24_root / "forward_sample_sufficiency.csv")
    risk = _read_csv(cfg.v24_root / "risk_envelope_check.csv")
    cost = _read_csv(cfg.v25_root / "execution_cost_stress.csv")
    assumptions = _read_csv(cfg.v25_root / "execution_assumption_check.csv")
    decisions = _read_csv(cfg.v24_root / "promotion_decision_table.csv")
    rows: list[dict[str, Any]] = []

    def add(check_id: str, observed: Any, required: Any, passed: bool, status: str, notes: str) -> None:
        rows.append(
            {
                "check_id": check_id,
                "observed": observed,
                "required": required,
                "passed": bool(passed),
                "status": status,
                "notes": notes,
            }
        )

    s5 = stack[stack.get("structure_id", pd.Series(dtype=str)).astype(str).eq("S5")]
    s5_suff = suff[suff.get("structure_id", pd.Series(dtype=str)).astype(str).eq("S5")]
    s5_risk = risk[risk.get("structure_id", pd.Series(dtype=str)).astype(str).eq("S5")]
    s5_cost = cost[cost.get("structure_id", pd.Series(dtype=str)).astype(str).eq("S5")]
    s5_decision = decisions[decisions.get("structure_id", pd.Series(dtype=str)).astype(str).eq("S5")]
    blockers = assumptions[assumptions.get("blocking_for_real_live", pd.Series(dtype=bool)).astype(bool)]
    max_exposure = float(_num(s5, "max_exposure").iloc[0]) if not s5.empty and _num(s5, "max_exposure").notna().any() else np.nan
    add(
        "sample_sufficiency",
        s5_suff["overall_sample_status"].iloc[0] if not s5_suff.empty else "missing",
        "sufficient",
        bool(not s5_suff.empty and s5_suff["overall_sample_status"].iloc[0] == "sufficient"),
        "not_ready",
        "Protect_A/O6/core forward samples must pass thresholds.",
    )
    add(
        "risk_envelope",
        s5_risk["risk_envelope_status"].iloc[0] if not s5_risk.empty else "missing",
        "pass",
        bool(not s5_risk.empty and s5_risk["risk_envelope_status"].iloc[0] == "pass"),
        "not_ready",
        "Requires net30/month_cap/worst-period checks.",
    )
    add(
        "cost_stress",
        s5_cost["cost_stress_status"].iloc[0] if not s5_cost.empty else "missing",
        "pass",
        bool(not s5_cost.empty and s5_cost["cost_stress_status"].iloc[0] == "pass"),
        "not_ready",
        "30bp must remain acceptable and 50bp must not collapse.",
    )
    add(
        "execution_blockers",
        int(len(blockers)),
        0,
        len(blockers) == 0,
        "not_ready" if len(blockers) else "pass",
        "v2.5 assumption blockers must be cleared.",
    )
    add(
        "total_exposure_cap",
        max_exposure,
        f"<= {cfg.total_exposure_cap}",
        bool(pd.isna(max_exposure) or max_exposure <= cfg.total_exposure_cap),
        "pass",
        "Current observed forward exposure is within proposed cap, but sample is small.",
    )
    add(
        "promotion_decision",
        s5_decision["decision"].iloc[0] if not s5_decision.empty else "missing",
        "PROMOTE_TO_SHADOW_CANDIDATE or better",
        bool(not s5_decision.empty and "PROMOTE" in str(s5_decision["decision"].iloc[0])),
        "not_ready",
        "v2.4 controls promotion status.",
    )
    return pd.DataFrame(rows)


def _notes_text(check: pd.DataFrame) -> str:
    passed = int(check["passed"].sum()) if not check.empty else 0
    total = int(len(check))
    ready = bool(total and passed == total)
    lines = [
        "# v2.6 Risk Envelope Finalization",
        "",
        "Status: draft risk envelope. No real-live permission is changed.",
        "",
        f"- checks_passed: {passed}/{total}",
        f"- canary_or_real_live_ready: {str(ready).lower()}",
        "",
        "## Decision",
    ]
    if ready:
        lines.append("- Risk envelope checks passed; proceed to canary readiness checklist.")
    else:
        lines.append("- Risk envelope is documented but not finalized for live promotion.")
        lines.append("- Continue forward paper/shadow logging and execution realism validation.")
    return "\n".join(lines) + "\n"


def _notes(root: Path, check: pd.DataFrame) -> None:
    text = _notes_text(check)
    root.joinpath("risk_envelope_decision.md").write_text(text, encoding="utf-8")
    root.joinpath("candidate_notes.md").write_text(text, encoding="utf-8")


def write_v26_risk_envelope_finalization(cfg: V26Config = V26Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    spec = _risk_policy_spec(cfg)
    stops = _stop_conditions(cfg)
    check = _current_envelope_check(cfg)
    outputs = {
        "risk_policy_spec": root / "risk_policy_spec.csv",
        "stop_conditions": root / "stop_conditions.csv",
        "current_envelope_check": root / "current_envelope_check.csv",
        "risk_envelope_decision": root / "risk_envelope_decision.md",
        "candidate_notes": root / "candidate_notes.md",
    }
    spec.to_csv(outputs["risk_policy_spec"], index=False)
    stops.to_csv(outputs["stop_conditions"], index=False)
    check.to_csv(outputs["current_envelope_check"], index=False)
    _notes(root, check)
    return outputs


__all__ = ["V26Config", "write_v26_risk_envelope_finalization"]
