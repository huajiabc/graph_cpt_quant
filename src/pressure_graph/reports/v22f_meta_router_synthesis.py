"""v2.2F meta-router synthesis and promotion decision."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v22a_meta_router_dataset_audit import REPORT_ROOT as V22A_ROOT
from pressure_graph.reports.v22b_preentry_meta_router import REPORT_ROOT as V22B_ROOT
from pressure_graph.reports.v22c_walkforward_policy_simulation import REPORT_ROOT as V22C_ROOT
from pressure_graph.reports.v22d_threshold_stability import REPORT_ROOT as V22D_ROOT
from pressure_graph.reports.v22e_negative_controls import REPORT_ROOT as V22E_ROOT


REPORT_ROOT = Path("reports/v2_2f_meta_router_synthesis")


@dataclass(frozen=True)
class V22FConfig:
    report_root: Path = REPORT_ROOT
    v22a_root: Path = V22A_ROOT
    v22b_root: Path = V22B_ROOT
    v22c_root: Path = V22C_ROOT
    v22d_root: Path = V22D_ROOT
    v22e_root: Path = V22E_ROOT


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _artifact_status(paths: list[Path]) -> tuple[str, str]:
    missing = [str(path) for path in paths if not path.exists()]
    return ("missing", ";".join(missing)) if missing else ("available", "")


def _stage_manifest(cfg: V22FConfig) -> pd.DataFrame:
    specs = [
        ("v2.2A", "Dataset Audit", cfg.v22a_root, ["router_head_trainability.csv", "feature_leakage_audit.csv"]),
        ("v2.2B", "Pre-entry Router", cfg.v22b_root, ["walkforward_policy_summary.csv", "negative_controls.csv"]),
        ("v2.2C", "Policy Simulation", cfg.v22c_root, ["policy_vs_benchmark.csv", "low_coimpulse_slice_summary.csv"]),
        ("v2.2D", "Threshold Stability", cfg.v22d_root, ["threshold_surface.csv", "threshold_plateau_summary.csv"]),
        ("v2.2E", "Negative Controls", cfg.v22e_root, ["negative_control_summary.csv"]),
    ]
    rows = []
    for stage, title, root, files in specs:
        status, missing = _artifact_status([root / file for file in files])
        rows.append(
            {
                "stage": stage,
                "title": title,
                "report_dir": str(root),
                "artifact_status": status,
                "missing_artifacts": missing,
            }
        )
    return pd.DataFrame(rows)


def _decision_matrix(cfg: V22FConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trainability = _read_csv(cfg.v22a_root / "router_head_trainability.csv")
    if not trainability.empty:
        for row in trainability.itertuples(index=False):
            rows.append(
                {
                    "item": f"{row.label_head}_trainability",
                    "status": row.trainability_status,
                    "evidence": (
                        f"active={row.active_or_decision_events}, "
                        f"min_class={row.min_active_label_events}, active_periods={row.active_periods}"
                    ),
                    "decision": "train_preentry_first" if "pre_entry" in row.label_head else "diagnostic_only",
                }
            )
    b = _read_csv(cfg.v22b_root / "walkforward_policy_summary.csv")
    if not b.empty:
        candidates = b[
            b["policy_id"].astype(str).str.startswith("logistic")
            & ~b["policy_id"].astype(str).str.contains("shuffled", regex=False)
        ].copy()
        if not candidates.empty:
            best = candidates.sort_values("validation_delta_vs_baseline_net20", ascending=False).iloc[0]
            rows.append(
                {
                    "item": "best_v22b_logistic_router",
                    "status": "weak_positive_not_shadow",
                    "evidence": (
                        f"{best.policy_id}: full_delta={best.delta_vs_baseline_net20:.4%}, "
                        f"validation_delta={best.validation_delta_vs_baseline_net20:.4%}, "
                        f"holdout_delta={best.holdout_delta_vs_baseline_net20:.4%}, "
                        f"random_pct={best.get('random_skip_match_t70_percentile', float('nan')):.2f}"
                    ),
                    "decision": "no_shadow",
                }
            )
    d = _read_csv(cfg.v22d_root / "threshold_plateau_summary.csv")
    if not d.empty:
        logistic = d[d["policy_family"].eq("logistic_no_trade")]
        if not logistic.empty:
            row = logistic.iloc[0]
            rows.append(
                {
                    "item": "logistic_threshold_plateau",
                    "status": row.stable_plateau_status,
                    "evidence": f"passing={row.passing_thresholds}/{row.thresholds_tested}; thresholds={row.passing_threshold_list}",
                    "decision": "needs_controls",
                }
            )
    e = _read_csv(cfg.v22e_root / "negative_control_summary.csv")
    if not e.empty:
        random = e[e["control_type"].eq("random_count_matched")]
        if not random.empty:
            row = random.iloc[0]
            status = "failed_random_p75" if float(row.primary_percentile_vs_control) < 0.75 else "passed_random_p75"
            rows.append(
                {
                    "item": "random_count_matched_control",
                    "status": status,
                    "evidence": (
                        f"median={row.delta_median:.4%}, p75={row.delta_p75:.4%}, "
                        f"p90={row.delta_p90:.4%}, primary_pct={row.primary_percentile_vs_control:.2f}"
                    ),
                    "decision": "no_shadow" if status == "failed_random_p75" else "further_audit",
                }
            )
    return pd.DataFrame(rows)


def _promotion_checklist(manifest: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    all_artifacts = bool(manifest["artifact_status"].eq("available").all()) if not manifest.empty else False
    random_pass = bool(decisions["status"].astype(str).eq("passed_random_p75").any()) if not decisions.empty else False
    validation_holdout_positive = False
    best = decisions[decisions["item"].eq("best_v22b_logistic_router")] if not decisions.empty else pd.DataFrame()
    if not best.empty:
        validation_holdout_positive = "validation_delta=" in str(best.iloc[0]["evidence"]) and "holdout_delta=" in str(best.iloc[0]["evidence"])
    checks = [
        ("all_artifacts_available", all_artifacts, "v2.2A-E reports exist"),
        ("preentry_trainability_confirmed", "pre_entry_action_label_trainability" in decisions["item"].tolist() if not decisions.empty else False, "v2.2A supports pre-entry first pass"),
        ("validation_holdout_positive_direction", validation_holdout_positive, "v2.2B has positive-direction evidence"),
        ("random_p75_control_passed", random_pass, "v2.2E random-count p75 must pass"),
        ("no_live_permission_change", True, "v2.2 remains offline research only"),
    ]
    return pd.DataFrame(
        [
            {
                "requirement": name,
                "status": "passed" if passed else "failed",
                "note": note,
            }
            for name, passed, note in checks
        ]
    )


def _notes(root: Path, manifest: pd.DataFrame, decisions: pd.DataFrame, checklist: pd.DataFrame) -> None:
    lines = [
        "# v2.2F Meta-router Synthesis",
        "",
        "Status: governance decision only. No shadow, paper-live, or real-live rule is promoted.",
        "",
        "## Stage Manifest",
    ]
    for row in manifest.itertuples(index=False):
        lines.append(f"- {row.stage} {row.title}: {row.artifact_status}")
    lines.extend(["", "## Decisions"])
    for row in decisions.itertuples(index=False):
        lines.append(f"- {row.item}: {row.status}; {row.evidence}; decision={row.decision}")
    lines.extend(["", "## Promotion Checklist"])
    for row in checklist.itertuples(index=False):
        lines.append(f"- {row.requirement}: {row.status}; {row.note}")
    if checklist["status"].eq("passed").all():
        lines.append("")
        lines.append("- All guardrails passed; candidate can move to deeper offline audit.")
    else:
        lines.append("")
        lines.append("- Guardrails are not fully passed. Keep v2.2 router offline/diagnostic; do not add shadow selector.")
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v22f_meta_router_synthesis(cfg: V22FConfig = V22FConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    manifest = _stage_manifest(cfg)
    decisions = _decision_matrix(cfg)
    checklist = _promotion_checklist(manifest, decisions)
    outputs = {
        "stage_manifest": root / "stage_manifest.csv",
        "decision_matrix": root / "decision_matrix.csv",
        "promotion_checklist": root / "promotion_checklist.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    manifest.to_csv(outputs["stage_manifest"], index=False)
    decisions.to_csv(outputs["decision_matrix"], index=False)
    checklist.to_csv(outputs["promotion_checklist"], index=False)
    _notes(root, manifest, decisions, checklist)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V22FConfig",
    "write_v22f_meta_router_synthesis",
]
