"""v2.1H State-discovery synthesis and v2.2 handoff.

Aggregates v2.0 / v2.1A-G report outputs into a compact decision manifest.
This report is a governance/handoff artifact only; it does not run a router,
change paper-live, or promote any live rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v2_1h_state_discovery_synthesis")


@dataclass(frozen=True)
class V21HConfig:
    report_root: Path = REPORT_ROOT
    reports_root: Path = Path("reports")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _pct(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(number) else f"{number:.4%}"


def _artifact_status(paths: list[Path]) -> tuple[str, str]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return "missing", ";".join(missing)
    return "available", ""


def _stage_manifest(root: Path) -> pd.DataFrame:
    specs = [
        (
            "v2.0",
            "Graph Motif Search",
            root / "v2_0_graph_motif_search",
            ["stable_candidates.csv", "top_paths.csv", "candidate_notes.md"],
            "No algorithm candidate promoted; ACO/GA/SA reproduced CIC/O6/Protect_A neighborhood but did not beat holdout benchmark.",
        ),
        (
            "v2.1A",
            "Holdout Autopsy",
            root / "v2_1a_holdout_autopsy",
            ["holdout_by_cic_type.csv", "candidate_notes.md"],
            "Holdout damage came from long continuation / CIC1 weakness, not skipped capacity or O6.",
        ),
        (
            "v2.1B",
            "State Cluster Atlas",
            root / "v2_1b_state_cluster_atlas",
            ["state_cluster_membership.csv", "state_cluster_summary.csv", "candidate_notes.md"],
            "State clusters identify weak holdout states but are diagnostic only.",
        ),
        (
            "v2.1C",
            "State Transition Graph",
            root / "v2_1c_state_transition_graph",
            ["state_transition_edges.csv", "holdout_transition_autopsy.csv", "candidate_notes.md"],
            "Transition damage is concentrated around CIC1/BTC_up flip and weak CIC2/BTC_chop states.",
        ),
        (
            "v2.1D",
            "Router Preflight",
            root / "v2_1d_router_preflight",
            ["router_rule_summary.csv", "candidate_notes.md"],
            "Holdout-derived filters can rescue damage but are leakage diagnostics; safe rules did not pass validation.",
        ),
        (
            "v2.1E",
            "Walk-forward Router Stability",
            root / "v2_1e_walkforward_router_stability",
            ["walkforward_router_summary.csv", "candidate_notes.md"],
            "Prior-only monthly routers failed validation/full walk-forward; no router shadow.",
        ),
        (
            "v2.1F",
            "State Drift Audit",
            root / "v2_1f_state_drift_audit",
            ["state_feature_period_drift.csv", "walkforward_novelty_bucket_summary.csv", "candidate_notes.md"],
            "Holdout has low market/cluster/burst/peer-count drift; novelty itself is not a simple risk-off rule.",
        ),
        (
            "v2.1G",
            "Meta-router Action Labels",
            root / "v2_1g_meta_router_action_labels",
            ["meta_router_event_labels.csv", "meta_router_feature_matrix.csv", "router_dataset_schema.csv", "candidate_notes.md"],
            "Offline action-label dataset is ready for v2.2; labels are realized counterfactuals, not as-of selectors.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for stage, title, report_dir, files, decision in specs:
        paths = [report_dir / file for file in files]
        status, missing = _artifact_status(paths)
        rows.append(
            {
                "stage": stage,
                "title": title,
                "report_dir": str(report_dir),
                "artifact_status": status,
                "missing_artifacts": missing,
                "decision": decision,
                "promoted_to_shadow": False,
                "promoted_to_paper_live": False,
                "real_live_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _decision_matrix(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    v20 = _read_csv(root / "v2_0_graph_motif_search" / "stable_candidates.csv")
    if not v20.empty:
        holdout_col = "holdout_portfolio_net20"
        validation_col = "validation_portfolio_net20"
        best = v20.sort_values("full_portfolio_net20", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            rows.append(
                {
                    "item": "v2.0_best_algorithm_architecture",
                    "status": "rejected_for_promotion",
                    "evidence": f"validation={_pct(row.get(validation_col))}, holdout={_pct(row.get(holdout_col))}",
                    "reason": "Algorithm search did not improve holdout robustness versus benchmark.",
                }
            )
    v21e = _read_csv(root / "v2_1e_walkforward_router_stability" / "walkforward_router_summary.csv")
    if not v21e.empty:
        candidates = v21e[~v21e["rule_id"].eq("baseline_B4")].copy()
        if not candidates.empty:
            best = candidates.sort_values("validation_delta_vs_baseline_net20", ascending=False).iloc[0]
            rows.append(
                {
                    "item": "prior_only_state_router",
                    "status": "rejected_for_shadow",
                    "evidence": (
                        f"best={best.rule_id}, validation_delta={_pct(best.validation_delta_vs_baseline_net20)}, "
                        f"holdout_delta={_pct(best.holdout_delta_vs_baseline_net20)}, "
                        f"full_delta={_pct(best.delta_vs_baseline_net20)}"
                    ),
                    "reason": "No prior-only router has positive validation and full walk-forward delta.",
                }
            )
    v21f = _read_csv(root / "v2_1f_state_drift_audit" / "state_feature_period_drift.csv")
    if not v21f.empty:
        holdout = v21f[v21f["period"].eq("holdout")].sort_values("psi_vs_preholdout", ascending=False)
        top = holdout.head(4)
        evidence = "; ".join(
            f"{row.feature}:psi={float(row.psi_vs_preholdout):.3f}" for row in top.itertuples(index=False)
        )
        rows.append(
            {
                "item": "holdout_state_drift",
                "status": "diagnostic_confirmed",
                "evidence": evidence,
                "reason": "Holdout is a low-coimpulse / low-burst / low-peer-count environment, but drift alone is not a rule.",
            }
        )
    v21g = _read_csv(root / "v2_1g_meta_router_action_labels" / "meta_router_event_labels.csv")
    if not v21g.empty:
        rows.append(
            {
                "item": "meta_router_label_dataset",
                "status": "ready_for_v2_2_offline_research",
                "evidence": (
                    f"events={len(v21g)}, pre_entry_labels="
                    f"{v21g['pre_entry_action_label'].value_counts().to_dict()}"
                ),
                "reason": "Feature matrix and multi-head realized-counterfactual labels exist for v2.2 learner evaluation.",
            }
        )
    return pd.DataFrame(rows)


def _readiness_checklist(root: Path, manifest: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    v21g_feature = root / "v2_1g_meta_router_action_labels" / "meta_router_feature_matrix.csv"
    v21g_schema = root / "v2_1g_meta_router_action_labels" / "router_dataset_schema.csv"
    v21e_summary = root / "v2_1e_walkforward_router_stability" / "walkforward_router_summary.csv"
    checks = [
        (
            "state_features_available",
            v21g_feature.exists(),
            str(v21g_feature),
            "v2.2 can read as-of state features from v2.1G feature matrix.",
        ),
        (
            "label_schema_available",
            v21g_schema.exists(),
            str(v21g_schema),
            "v2.2 labels are documented with roles and label_source.",
        ),
        (
            "prior_router_rejected",
            "prior_only_state_router" in decisions.get("item", pd.Series(dtype=str)).tolist(),
            str(v21e_summary),
            "v2.2 should not inherit v2.1D/E hard router rules as promoted candidates.",
        ),
        (
            "all_stage_artifacts_available",
            bool(manifest["artifact_status"].eq("available").all()) if not manifest.empty else False,
            "v2.0 and v2.1A-G report roots",
            "Synthesis can trace every prior decision to a concrete report artifact.",
        ),
        (
            "no_live_promotion",
            bool((~manifest["promoted_to_paper_live"]).all() and (~manifest["real_live_allowed"]).all()) if not manifest.empty else False,
            "stage_manifest.csv",
            "v2.1 remains offline diagnostic/research; no live permissions are changed.",
        ),
    ]
    rows = []
    for requirement, passed, evidence, note in checks:
        rows.append(
            {
                "requirement": requirement,
                "status": "passed" if passed else "failed",
                "evidence": evidence,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def _notes(root: Path, manifest: pd.DataFrame, decisions: pd.DataFrame, readiness: pd.DataFrame) -> None:
    lines = [
        "# v2.1H State Discovery Synthesis",
        "",
        "Status: governance / handoff report only. No live, paper-live, or shadow rule is promoted.",
        "",
        "## Stage Decisions",
    ]
    for row in manifest.itertuples(index=False):
        lines.append(f"- {row.stage} {row.title}: {row.decision}")
    lines.extend(["", "## Decision Matrix"])
    for row in decisions.itertuples(index=False):
        lines.append(f"- {row.item}: {row.status}; {row.evidence}; {row.reason}")
    lines.extend(["", "## v2.2 Readiness"])
    for row in readiness.itertuples(index=False):
        lines.append(f"- {row.requirement}: {row.status}; {row.note}")
    passed = readiness["status"].eq("passed").all() if not readiness.empty else False
    lines.extend(["", "## Handoff"])
    if passed:
        lines.append(
            "- v2.2 can start as an offline walk-forward meta-router learner using v2.1G features/labels. "
            "Any candidate must beat B4/P2+O6+Protect_A benchmarks out of sample before shadow discussion."
        )
    else:
        lines.append("- v2.2 handoff is incomplete; inspect failed readiness rows before proceeding.")
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21h_state_discovery_synthesis(cfg: V21HConfig = V21HConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    report_root = cfg.reports_root
    manifest = _stage_manifest(report_root)
    decisions = _decision_matrix(report_root)
    readiness = _readiness_checklist(report_root, manifest, decisions)

    outputs = {
        "stage_manifest": root / "stage_manifest.csv",
        "decision_matrix": root / "decision_matrix.csv",
        "v22_readiness_checklist": root / "v22_readiness_checklist.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    manifest.to_csv(outputs["stage_manifest"], index=False)
    decisions.to_csv(outputs["decision_matrix"], index=False)
    readiness.to_csv(outputs["v22_readiness_checklist"], index=False)
    _notes(root, manifest, decisions, readiness)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V21HConfig",
    "write_v21h_state_discovery_synthesis",
]
