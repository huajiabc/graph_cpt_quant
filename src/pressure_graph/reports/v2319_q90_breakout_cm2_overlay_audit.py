"""Independent artifact audit of the frozen v23.18 CM2 overlay reveal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2317_q90_breakout_cm2_feature_audit import (
    REPORT_ROOT as V2317_ROOT,
    V2317Config,
    audit_v2317,
    build_v2317_event_mapping,
    feature_hash_v2317,
    load_v2317_inputs,
    summarize_v2317,
)
from pressure_graph.reports.v2318_q90_breakout_cm2_overlay import (
    FROZEN_FEATURE_HASH,
    REPORT_ROOT as V2318_ROOT,
    V2318Config,
    build_v2318_leave_one_month_out,
    build_v2318_month_bootstrap,
    build_v2318_panel,
    build_v2318_sensitivity,
    evaluate_v2318_gates,
    load_v2318_inputs,
    summarize_v2318,
)


REPORT_ROOT = Path("reports/v23_19_q90_breakout_cm2_overlay_audit")
FINDINGS_PATH = Path("docs/v2319_q90_breakout_cm2_overlay_audit_2026_07_17.md")


@dataclass(frozen=True)
class V2319Config:
    v2317_root: Path = V2317_ROOT
    v2318_root: Path = V2318_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    tolerance: float = 1e-12


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame, tolerance: float) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=tolerance,
        )
    except AssertionError:
        return False
    return True


def _maximum_numeric_error(left: pd.DataFrame, right: pd.DataFrame) -> float:
    shared = [
        column
        for column in left.columns.intersection(right.columns)
        if pd.api.types.is_numeric_dtype(left[column])
        and pd.api.types.is_numeric_dtype(right[column])
    ]
    errors = []
    for column in shared:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if len(a) != len(b):
            return np.inf
        finite = np.isfinite(a) & np.isfinite(b)
        errors.extend(np.abs(a[finite] - b[finite]).tolist())
        errors.extend([np.inf] * int(np.sum(np.isfinite(a) != np.isfinite(b))))
    return float(max(errors, default=0.0))


def run_v2319_audit(cfg: V2319Config = V2319Config()) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = V2317Config(report_root=cfg.v2317_root)
    raw_features, raw_calendar = load_v2317_inputs(feature_cfg)
    rebuilt_mapping = build_v2317_event_mapping(raw_features, raw_calendar, feature_cfg)
    rebuilt_feature_summary = summarize_v2317(rebuilt_mapping, raw_calendar)
    rebuilt_feature_checks = audit_v2317(
        rebuilt_mapping,
        raw_calendar,
        rebuilt_feature_summary,
        feature_cfg,
    )
    saved_mapping = pd.read_parquet(cfg.v2317_root / "q90_event_week_mapping.parquet")
    saved_feature_summary = pd.read_csv(cfg.v2317_root / "feature_coverage_summary.csv")
    saved_feature_checks = pd.read_csv(cfg.v2317_root / "data_quality_checks.csv")

    reveal_cfg = V2318Config(v2317_root=cfg.v2317_root, report_root=cfg.v2318_root)
    mapping, outcomes, core = load_v2318_inputs(reveal_cfg)
    rebuilt_events, rebuilt_panel = build_v2318_panel(mapping, outcomes, core, reveal_cfg)
    rebuilt_summary = summarize_v2318(rebuilt_panel)
    rebuilt_sensitivity = build_v2318_sensitivity(rebuilt_panel, reveal_cfg)
    rebuilt_bootstrap = build_v2318_month_bootstrap(rebuilt_panel, reveal_cfg)
    rebuilt_leaveout = build_v2318_leave_one_month_out(rebuilt_panel)
    rebuilt_gates = evaluate_v2318_gates(
        mapping,
        rebuilt_events,
        rebuilt_panel,
        rebuilt_summary,
        rebuilt_sensitivity,
        rebuilt_bootstrap,
        rebuilt_leaveout,
    )
    saved_events = pd.read_parquet(cfg.v2318_root / "mapped_event_outcomes.parquet")
    saved_panel = pd.read_parquet(cfg.v2318_root / "weekly_portfolio.parquet")
    saved_summary = pd.read_csv(cfg.v2318_root / "summary.csv")
    saved_sensitivity = pd.read_csv(cfg.v2318_root / "allocation_sensitivity.csv")
    saved_bootstrap = pd.read_parquet(cfg.v2318_root / "month_bootstrap.parquet")
    saved_leaveout = pd.read_csv(cfg.v2318_root / "leave_one_month_out.csv")
    saved_leaveout["excluded_month"] = pd.to_datetime(
        saved_leaveout["excluded_month"], utc=True
    )
    saved_gates = pd.read_csv(cfg.v2318_root / "evidence_gates.csv")
    metadata = json.loads((cfg.v2318_root / "metadata.json").read_text(encoding="utf-8"))

    comparisons = {
        "feature_mapping": (rebuilt_mapping, saved_mapping),
        "feature_summary": (rebuilt_feature_summary, saved_feature_summary),
        "feature_checks": (rebuilt_feature_checks, saved_feature_checks),
        "event_outcomes": (rebuilt_events, saved_events),
        "weekly_portfolio": (rebuilt_panel, saved_panel),
        "summary": (rebuilt_summary, saved_summary),
        "sensitivity": (rebuilt_sensitivity, saved_sensitivity),
        "bootstrap": (rebuilt_bootstrap, saved_bootstrap),
        "leave_one_month_out": (rebuilt_leaveout, saved_leaveout),
        "evidence_gates": (rebuilt_gates, saved_gates),
    }
    diagnostics = pd.DataFrame(
        [
            {
                "artifact": artifact,
                "rows_rebuilt": len(left),
                "rows_saved": len(right),
                "maximum_numeric_error": _maximum_numeric_error(left, right),
                "exact_within_tolerance": _frames_equal(left, right, cfg.tolerance),
            }
            for artifact, (left, right) in comparisons.items()
        ]
    )
    failed_gates = saved_gates.loc[~saved_gates["passed"], "gate"].tolist()
    checks = {
        "v2317_feature_audit_rebuilds": rebuilt_feature_checks["passed"].all(),
        "frozen_feature_hash_exact": feature_hash_v2317(rebuilt_mapping)
        == FROZEN_FEATURE_HASH,
        "all_artifacts_exact_within_tolerance": diagnostics[
            "exact_within_tolerance"
        ].all(),
        "all_numeric_errors_within_tolerance": diagnostics[
            "maximum_numeric_error"
        ].le(cfg.tolerance).all(),
        "exactly_53_events_and_49_weeks": len(rebuilt_events) == 53
        and len(rebuilt_panel) == 49,
        "all_events_triggered": int(rebuilt_events["triggered"].sum()) == 53,
        "frozen_overlay_weight_exact": np.isclose(
            rebuilt_panel["overlay_weight"], reveal_cfg.overlay_weight
        ).all(),
        "cross_week_events_use_realization_clock": (
            rebuilt_mapping.loc[rebuilt_mapping["crosses_calendar_week"], "event_exit_time"]
            .ge(
                rebuilt_mapping.loc[
                    rebuilt_mapping["crosses_calendar_week"], "portfolio_entry_time"
                ]
            )
            .all()
        ),
        "only_bootstrap_lower_gate_failed": failed_gates
        == ["absolute_month_bootstrap_lower_above_zero"],
        "rejection_verdict_exact": metadata["verdict"]
        == "q90_cm2_portfolio_confirmation_rejected"
        and not metadata["all_gates_passed"],
    }
    audit_checks = pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )
    return audit_checks, diagnostics


def write_v2319_q90_breakout_cm2_overlay_audit(
    cfg: V2319Config = V2319Config(),
) -> dict[str, Path]:
    checks, diagnostics = run_v2319_audit(cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "checks": root / "audit_checks.csv",
        "diagnostics": root / "audit_diagnostics.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    checks.to_csv(paths["checks"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    passed = bool(checks["passed"].all())
    paths["metadata"].write_text(
        json.dumps(
            {
                "audit_passed": passed,
                "checks_passed": int(checks["passed"].sum()),
                "checks_total": len(checks),
                "tolerance": cfg.tolerance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.19 q90 Breakout + CM2 Overlay Independent Audit",
                "",
                f"Verdict: `{'audit_passed' if passed else 'audit_failed'}`.",
                "",
                checks.to_markdown(index=False),
                "",
                diagnostics.to_markdown(index=False, floatfmt=".4g"),
                "",
                "The audit reproduces the feature clock, realization-week mapping,",
                "event compounding, portfolio metrics, bootstrap, gates, and rejection",
                "decision independently from the saved v23.18 artifacts.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["V2319Config", "run_v2319_audit", "write_v2319_q90_breakout_cm2_overlay_audit"]
