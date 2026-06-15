"""v2.2A meta-router dataset audit.

Audits the v2.1G realized-counterfactual label dataset before any model is
trained. This report is intentionally descriptive: it does not fit a router,
create a selector, or change live/shadow permissions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v21g_meta_router_action_labels import STATE_COLUMNS


REPORT_ROOT = Path("reports/v2_2a_meta_router_dataset_audit")
V21G_ROOT = Path("reports/v2_1g_meta_router_action_labels")
LABEL_HEADS = [
    "pre_entry_action_label",
    "post_entry_checkpoint_label",
    "protect_a_label",
    "capacity_overflow_label",
]
PRIMARY_FEATURE_BLOCKLIST = {
    "state_cluster_id",
    "novelty_bucket",
    "walkforward_state_novelty",
}
COUNTERFACTUAL_PATTERNS = (
    "net20",
    "utility_",
    "_label",
    "_delta_",
    "future",
    "mfe_",
    "mae_",
    "hit_",
    "exit_",
)


@dataclass(frozen=True)
class V22AConfig:
    report_root: Path = REPORT_ROOT
    v21g_root: Path = V21G_ROOT
    min_train_events: int = 100
    min_class_events: int = 20
    min_active_periods: int = 2


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _role_map(schema: pd.DataFrame) -> dict[str, str]:
    if schema.empty or "column" not in schema.columns or "role" not in schema.columns:
        return {}
    return dict(zip(schema["column"].astype(str), schema["role"].astype(str), strict=False))


def _label_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(frame), 1)
    for head in LABEL_HEADS:
        if head not in frame.columns:
            continue
        for label, group in frame.groupby(head, dropna=False, sort=False):
            net = _num(group, "net20")
            rows.append(
                {
                    "label_head": head,
                    "label": str(label),
                    "events": int(len(group)),
                    "share": float(len(group) / total),
                    "net20_sum": float(net.sum()),
                    "net20_avg": float(net.mean()) if len(group) else np.nan,
                    "hit_rate": float(net.gt(0).mean()) if len(group) else np.nan,
                    "search_events": int(group.get("period", pd.Series(dtype=str)).eq("search").sum()),
                    "validation_events": int(group.get("period", pd.Series(dtype=str)).eq("validation").sum()),
                    "holdout_events": int(group.get("period", pd.Series(dtype=str)).eq("holdout").sum()),
                    "months": int(group.get("entry_month", pd.Series(dtype=str)).nunique()),
                }
            )
    return pd.DataFrame(rows)


def _label_by_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_col not in frame.columns:
        return pd.DataFrame()
    for head in LABEL_HEADS:
        if head not in frame.columns:
            continue
        grouped = frame.groupby([group_col, head], dropna=False, sort=False)
        for keys, group in grouped:
            group_value, label = keys
            net = _num(group, "net20")
            rows.append(
                {
                    group_col: group_value,
                    "label_head": head,
                    "label": str(label),
                    "events": int(len(group)),
                    "net20_sum": float(net.sum()),
                    "net20_avg": float(net.mean()) if len(group) else np.nan,
                    "hit_rate": float(net.gt(0).mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _feature_coverage(frame: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in frame.columns:
        series = frame[col]
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        role = roles.get(col, "unknown")
        is_state = col in STATE_COLUMNS
        ablation_only = col in PRIMARY_FEATURE_BLOCKLIST
        allowed_primary = is_state and not ablation_only
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "schema_role": role,
                "non_null": non_null,
                "missing_rate": float(1.0 - non_null / max(len(frame), 1)),
                "unique_values": int(series.nunique(dropna=True)),
                "numeric_non_null": int(numeric.notna().sum()),
                "numeric_mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "numeric_median": float(numeric.median()) if numeric.notna().any() else np.nan,
                "numeric_min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "numeric_max": float(numeric.max()) if numeric.notna().any() else np.nan,
                "asof_state_feature": bool(is_state),
                "primary_router_feature_allowed": bool(allowed_primary),
                "ablation_only_feature": bool(ablation_only),
            }
        )
    return pd.DataFrame(rows)


def _feature_leakage_audit(frame: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in frame.columns:
        role = roles.get(col, "unknown")
        lower = col.lower()
        matched = [pattern for pattern in COUNTERFACTUAL_PATTERNS if pattern in lower]
        if role == "feature" and col not in PRIMARY_FEATURE_BLOCKLIST:
            status = "allowed_primary_asof_feature"
            issue = ""
        elif col in PRIMARY_FEATURE_BLOCKLIST:
            status = "ablation_only_high_overfit_risk"
            issue = "state/novelty derived feature should be tested separately from raw feature primary model"
        elif role == "label":
            status = "blocked_label"
            issue = "realized counterfactual label, never a model feature"
        elif role == "counterfactual_utility" or matched:
            status = "blocked_counterfactual_or_realized_outcome"
            issue = "contains realized utility/outcome semantics"
        else:
            status = "metadata_not_model_feature"
            issue = "identifier, split, timestamp, or descriptive metadata"
        rows.append(
            {
                "column": col,
                "schema_role": role,
                "matched_risk_patterns": ",".join(matched),
                "audit_status": status,
                "issue": issue,
                "use_in_primary_v22_model": status == "allowed_primary_asof_feature",
            }
        )
    return pd.DataFrame(rows)


def _head_trainability(frame: pd.DataFrame, distribution: pd.DataFrame, cfg: V22AConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for head in LABEL_HEADS:
        if head not in frame.columns:
            continue
        counts = frame[head].value_counts(dropna=False)
        if head == "pre_entry_action_label":
            active = counts[counts.index.astype(str).isin(["core_trade", "no_trade"])]
        elif head == "post_entry_checkpoint_label":
            active = counts[counts.index.astype(str).isin(["cp60_exit_better", "keep_better_false_exit_risk"])]
        elif head == "capacity_overflow_label":
            active = counts[counts.index.astype(str).isin(["allow_overflow", "deny_overflow"])]
        elif head == "protect_a_label":
            active = counts[counts.index.astype(str).isin(["protect_keep_better", "cp60_exit_better"])]
        else:
            active = counts[~counts.index.astype(str).str.startswith("not_")]
        eligible = int(active.sum())
        min_active = int(active.min()) if not active.empty else 0
        active_periods = int(
            frame.loc[frame[head].astype(str).isin(active.index.astype(str)), "period"].nunique()
            if "period" in frame.columns
            else 0
        )
        top_label_share = float(counts.max() / max(counts.sum(), 1)) if not counts.empty else np.nan
        label_rows = distribution[distribution["label_head"].eq(head)].copy()
        top_month_share = np.nan
        if not label_rows.empty and "entry_month" in frame.columns:
            month_counts = frame.loc[
                frame[head].astype(str).str.startswith("not_").eq(False), "entry_month"
            ].value_counts()
            if not month_counts.empty:
                top_month_share = float(month_counts.max() / max(month_counts.sum(), 1))
        if head == "pre_entry_action_label" and len(frame) >= cfg.min_train_events and min_active >= cfg.min_class_events:
            status = "trainable_first_pass_binary_core_vs_no_trade"
            reason = (
                "core_trade/no_trade support is sufficient; small_or_neutral should be neutral or reduced-size "
                "fallback, not a first-pass standalone class"
            )
        elif eligible < cfg.min_train_events or min_active < cfg.min_class_events:
            status = "diagnostic_only_small_or_imbalanced"
            reason = "active labels are too sparse or class support is too imbalanced for promotion-grade training"
        elif active_periods < cfg.min_active_periods:
            status = "diagnostic_only_period_concentrated"
            reason = "active labels are concentrated in too few periods"
        else:
            status = "trainable_diagnostic_first"
            reason = "label head can be explored, but should not be prioritized before pre-entry router"
        rows.append(
            {
                "label_head": head,
                "events": int(len(frame)),
                "active_or_decision_events": eligible,
                "min_active_label_events": min_active,
                "active_periods": active_periods,
                "top_label_share": top_label_share,
                "top_active_month_share": top_month_share,
                "trainability_status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _notes(root: Path, frame: pd.DataFrame, trainability: pd.DataFrame, leakage: pd.DataFrame) -> None:
    lines = [
        "# v2.2A Meta-router Dataset Audit",
        "",
        "Status: dataset audit only. No model, router, selector, shadow, paper-live, or real-live rule is promoted.",
        "",
        f"Events: {len(frame)}",
        "",
        "## Trainability",
    ]
    for row in trainability.itertuples(index=False):
        lines.append(
            f"- {row.label_head}: {row.trainability_status}; "
            f"active_events={row.active_or_decision_events}, "
            f"min_active_label_events={row.min_active_label_events}, "
            f"active_periods={row.active_periods}. {row.reason}"
        )
    allowed = leakage[leakage["use_in_primary_v22_model"].eq(True)] if not leakage.empty else pd.DataFrame()
    ablation = leakage[leakage["audit_status"].eq("ablation_only_high_overfit_risk")] if not leakage.empty else pd.DataFrame()
    lines.extend(
        [
            "",
            "## Feature Policy",
            f"- Primary as-of feature count: {len(allowed)}.",
            "- Labels, utilities, realized net/delta columns, and IDs are blocked from model features.",
            f"- Ablation-only high-overfit-risk features: {', '.join(ablation['column'].tolist()) if not ablation.empty else 'none'}.",
            "",
            "## Decision",
            "- Proceed next with a simple walk-forward pre-entry router only.",
            "- Checkpoint/protect/overflow heads remain diagnostic until sample support improves.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v22a_meta_router_dataset_audit(cfg: V22AConfig = V22AConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    feature_matrix = _read_csv(cfg.v21g_root / "meta_router_feature_matrix.csv")
    schema = _read_csv(cfg.v21g_root / "router_dataset_schema.csv")
    if feature_matrix.empty:
        raise FileNotFoundError(f"Missing or empty v2.1G feature matrix under {cfg.v21g_root}")
    roles = _role_map(schema)

    distribution = _label_distribution(feature_matrix)
    coverage = _feature_coverage(feature_matrix, roles)
    leakage = _feature_leakage_audit(feature_matrix, roles)
    trainability = _head_trainability(feature_matrix, distribution, cfg)

    outputs = {
        "label_distribution": root / "label_distribution.csv",
        "feature_coverage": root / "feature_coverage.csv",
        "feature_leakage_audit": root / "feature_leakage_audit.csv",
        "label_by_month": root / "label_by_month.csv",
        "label_by_cic_type": root / "label_by_cic_type.csv",
        "label_by_state_cluster": root / "label_by_state_cluster.csv",
        "label_by_market_regime": root / "label_by_market_regime.csv",
        "router_head_trainability": root / "router_head_trainability.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    distribution.to_csv(outputs["label_distribution"], index=False)
    coverage.to_csv(outputs["feature_coverage"], index=False)
    leakage.to_csv(outputs["feature_leakage_audit"], index=False)
    _label_by_group(feature_matrix, "entry_month").to_csv(outputs["label_by_month"], index=False)
    _label_by_group(feature_matrix, "cic_type").to_csv(outputs["label_by_cic_type"], index=False)
    _label_by_group(feature_matrix, "state_cluster_id").to_csv(outputs["label_by_state_cluster"], index=False)
    _label_by_group(feature_matrix, "btc_state").to_csv(outputs["label_by_market_regime"], index=False)
    trainability.to_csv(outputs["router_head_trainability"], index=False)
    _notes(root, feature_matrix, trainability, leakage)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V22AConfig",
    "write_v22a_meta_router_dataset_audit",
]
