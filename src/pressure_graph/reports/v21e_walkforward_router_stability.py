"""v2.1E Walk-forward router stability audit.

Derives state-cluster / transition router targets from prior months only and
applies them to the next month. This is an offline leakage-control report, not
a live router.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import V20Config, _prepare_sample, _simulate_portfolio
from pressure_graph.reports.v21b_state_cluster_atlas import V21BConfig, _prepare_membership
from pressure_graph.reports.v21d_router_preflight import BASE_SPEC, _add_transition_keys, _ledger_metrics


REPORT_ROOT = Path("reports/v2_1e_walkforward_router_stability")
TARGET_SEP = ";;"


@dataclass(frozen=True)
class V21EConfig:
    report_root: Path = REPORT_ROOT
    v21b: V21BConfig = V21BConfig(v20=V20Config())
    reduce_size_multiplier: float = 0.5
    min_cluster_events: int = 5
    min_transition_events: int = 3
    min_history_months: int = 2


def _entry_month(frame: pd.DataFrame) -> pd.Series:
    entry = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    return entry.dt.strftime("%Y-%m")


def _period_from_month(month: str) -> str:
    if month < "2026-02":
        return "search"
    if month < "2026-05":
        return "validation"
    return "holdout"


def _target_join(values: list[str]) -> str:
    return TARGET_SEP.join(sorted(set(value for value in values if value)))


def _target_split(value: object) -> list[str]:
    text = str(value or "")
    if not text:
        return []
    return [item for item in text.split(TARGET_SEP) if item]


def _bad_clusters(history: pd.DataFrame, cfg: V21EConfig) -> list[str]:
    if history.empty:
        return []
    grouped = (
        history.groupby("state_cluster_id", sort=False)["net20"]
        .agg(["count", "sum"])
        .reset_index()
    )
    bad = grouped[grouped["count"].ge(cfg.min_cluster_events) & grouped["sum"].lt(0)]
    return bad["state_cluster_id"].astype(str).tolist()


def _bad_transitions(history: pd.DataFrame, cfg: V21EConfig) -> list[str]:
    if history.empty:
        return []
    rows: list[dict[str, Any]] = []
    for col in ("global_h1_key", "same_symbol_h1_key", "same_burst_h1_key"):
        valid = history[history.get(col, pd.Series("", index=history.index)).astype(str).ne("")]
        if valid.empty:
            continue
        grouped = valid.groupby(col, sort=False)["net20"].agg(["count", "sum"]).reset_index()
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "transition_key": str(getattr(row, col)),
                    "count": int(row.count),
                    "sum": float(row.sum),
                }
            )
    if not rows:
        return []
    stats = pd.DataFrame(rows).groupby("transition_key", sort=False).agg({"count": "sum", "sum": "sum"}).reset_index()
    bad = stats[stats["count"].ge(cfg.min_transition_events) & stats["sum"].lt(0)]
    return bad["transition_key"].astype(str).tolist()


def _build_month_rules(history: pd.DataFrame, month: str, cfg: V21EConfig) -> pd.DataFrame:
    history_months = sorted(history["entry_month"].dropna().astype(str).unique().tolist()) if not history.empty else []
    enough_history = len(history_months) >= cfg.min_history_months
    cluster_targets = _bad_clusters(history, cfg) if enough_history else []
    transition_targets = _bad_transitions(history, cfg) if enough_history else []
    target_specs = [
        ("prior_bad_clusters", "cluster", cluster_targets),
        ("prior_bad_transitions", "transition", transition_targets),
        ("prior_bad_cluster_or_transition", "cluster_or_transition", [*cluster_targets, *transition_targets]),
    ]
    rows: list[dict[str, Any]] = [
        {
            "eval_month": month,
            "rule_id": "baseline_B4",
            "rule_type": "baseline",
            "action": "none",
            "source": "benchmark",
            "target_values": "",
            "target_count": 0,
            "history_months": len(history_months),
            "history_start_month": history_months[0] if history_months else "",
            "history_end_month": history_months[-1] if history_months else "",
            "rule_available": True,
            "leakage_status": "walkforward_prior_only",
        }
    ]
    for source, rule_type, targets in target_specs:
        targets = sorted(set(str(target) for target in targets if str(target)))
        for action in ("no_trade", "reduce_size_50"):
            rows.append(
                {
                    "eval_month": month,
                    "rule_id": f"{action}_{source}",
                    "rule_type": rule_type,
                    "action": action,
                    "source": source,
                    "target_values": _target_join(targets),
                    "target_count": len(targets),
                    "history_months": len(history_months),
                    "history_start_month": history_months[0] if history_months else "",
                    "history_end_month": history_months[-1] if history_months else "",
                    "rule_available": enough_history and bool(targets),
                    "leakage_status": "walkforward_prior_only",
                }
            )
    return pd.DataFrame(rows)


def _rule_mask(sample: pd.DataFrame, rule: pd.Series) -> pd.Series:
    targets = _target_split(rule.get("target_values", ""))
    if not targets or rule.get("rule_type") == "baseline":
        return pd.Series(False, index=sample.index)
    rule_type = str(rule.get("rule_type"))
    cluster_mask = sample["state_cluster_id"].astype(str).isin(targets)
    transition_mask = pd.Series(False, index=sample.index)
    for col in ("global_h1_key", "same_symbol_h1_key", "same_burst_h1_key"):
        if col in sample.columns:
            transition_mask = transition_mask | sample[col].astype(str).isin(targets)
    if rule_type == "cluster":
        return cluster_mask
    if rule_type == "transition":
        return transition_mask
    if rule_type == "cluster_or_transition":
        return cluster_mask | transition_mask
    return pd.Series(False, index=sample.index)


def _simulate_rule(sample: pd.DataFrame, rule: pd.Series, cfg: V21EConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mask = _rule_mask(sample, rule)
    action = str(rule.get("action", "none"))
    router_affected = 0
    if action == "no_trade":
        router_affected = int(mask.sum())
        ledger, skipped = _simulate_portfolio(sample[~mask].copy(), BASE_SPEC)
    else:
        ledger, skipped = _simulate_portfolio(sample, BASE_SPEC)
        if action == "reduce_size_50" and not ledger.empty:
            selected_mask = _rule_mask(ledger, rule)
            router_affected = int(selected_mask.sum())
            ledger = ledger.copy()
            ledger["router_reduced_size"] = selected_mask
            ledger.loc[selected_mask, "weighted_return"] = (
                pd.to_numeric(ledger.loc[selected_mask, "weighted_return"], errors="coerce")
                * float(cfg.reduce_size_multiplier)
            )
    metrics = _ledger_metrics(ledger, BASE_SPEC.max_positions)
    metrics.update(
        {
            "router_affected_events": router_affected,
            "skipped_trades": int(len(skipped)),
            "rule_match_events": int(mask.sum()),
        }
    )
    return ledger, skipped, metrics


def _month_cap(values: pd.Series, cap: float = 0.35) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    total = float(numeric.sum())
    cap_value = total * cap if total > 0 else 0.0
    capped = [min(value, cap_value) if value > 0 and cap_value > 0 else value for value in numeric]
    return float(np.sum(capped))


def _aggregate_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if monthly.empty:
        return pd.DataFrame()
    data = monthly.copy()
    if "baseline_month_net20" not in data.columns:
        baseline = data[data["rule_id"].eq("baseline_B4")][["eval_month", "portfolio_net20"]].rename(
            columns={"portfolio_net20": "baseline_month_net20"}
        )
        data = data.merge(baseline, on="eval_month", how="left")
    if "delta_vs_baseline_net20" not in data.columns:
        data["delta_vs_baseline_net20"] = (
            pd.to_numeric(data["portfolio_net20"], errors="coerce")
            - pd.to_numeric(data["baseline_month_net20"], errors="coerce")
        )
    for rule_id, group in data.groupby("rule_id", sort=False):
        row: dict[str, Any] = {
            "rule_id": rule_id,
            "months": int(group["eval_month"].nunique()),
            "applied_months": int(group["rule_match_events"].gt(0).sum()),
            "available_months": int(group["rule_available"].fillna(False).astype(bool).sum()),
            "selected_trades": int(pd.to_numeric(group["selected_trades"], errors="coerce").sum()),
            "skipped_trades": int(pd.to_numeric(group["skipped_trades"], errors="coerce").sum()),
            "router_affected_events": int(pd.to_numeric(group["router_affected_events"], errors="coerce").sum()),
            "portfolio_net20": float(pd.to_numeric(group["portfolio_net20"], errors="coerce").sum()),
            "baseline_net20": float(pd.to_numeric(group["baseline_month_net20"], errors="coerce").sum()),
            "delta_vs_baseline_net20": float(pd.to_numeric(group["delta_vs_baseline_net20"], errors="coerce").sum()),
            "avg_month_delta_net20": float(pd.to_numeric(group["delta_vs_baseline_net20"], errors="coerce").mean()),
            "months_improved": int(pd.to_numeric(group["delta_vs_baseline_net20"], errors="coerce").gt(0).sum()),
            "months_worse": int(pd.to_numeric(group["delta_vs_baseline_net20"], errors="coerce").lt(0).sum()),
            "month_cap35_net20": _month_cap(group["portfolio_net20"]),
            "worst_month_net20": float(pd.to_numeric(group["portfolio_net20"], errors="coerce").min()),
            "worst_month_delta": float(pd.to_numeric(group["delta_vs_baseline_net20"], errors="coerce").min()),
        }
        for period in ("search", "validation", "holdout"):
            part = group[group["period"].eq(period)]
            row[f"{period}_months"] = int(part["eval_month"].nunique())
            row[f"{period}_portfolio_net20"] = float(pd.to_numeric(part["portfolio_net20"], errors="coerce").sum())
            row[f"{period}_baseline_net20"] = float(pd.to_numeric(part["baseline_month_net20"], errors="coerce").sum())
            row[f"{period}_delta_vs_baseline_net20"] = float(pd.to_numeric(part["delta_vs_baseline_net20"], errors="coerce").sum())
            row[f"{period}_months_improved"] = int(pd.to_numeric(part["delta_vs_baseline_net20"], errors="coerce").gt(0).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["validation_delta_vs_baseline_net20", "holdout_delta_vs_baseline_net20", "delta_vs_baseline_net20"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def evaluate_walkforward_router(membership: pd.DataFrame, cfg: V21EConfig = V21EConfig()) -> dict[str, pd.DataFrame]:
    data = _add_transition_keys(membership)
    data["entry_month"] = _entry_month(data)
    data = data.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)
    months = sorted(data["entry_month"].dropna().astype(str).unique().tolist())

    monthly_rows: list[dict[str, Any]] = []
    catalog_rows: list[pd.DataFrame] = []
    ledgers: list[pd.DataFrame] = []
    skipped_rows: list[pd.DataFrame] = []
    for month in months:
        current = data[data["entry_month"].eq(month)].copy()
        history = data[data["entry_month"].lt(month)].copy()
        rules = _build_month_rules(history, month, cfg)
        catalog_rows.append(rules.copy())
        for _, rule in rules.iterrows():
            ledger, skipped, metrics = _simulate_rule(current, rule, cfg)
            row = rule.to_dict()
            row["period"] = _period_from_month(month)
            row.update(metrics)
            monthly_rows.append(row)
            if not ledger.empty:
                ledger = ledger.copy()
                ledger["eval_month"] = month
                ledger["rule_id"] = rule["rule_id"]
                ledgers.append(ledger)
            if not skipped.empty:
                skipped = skipped.copy()
                skipped["eval_month"] = month
                skipped["rule_id"] = rule["rule_id"]
                skipped_rows.append(skipped)

    monthly = pd.DataFrame(monthly_rows)
    if not monthly.empty:
        baseline = monthly[monthly["rule_id"].eq("baseline_B4")][["eval_month", "portfolio_net20"]].rename(
            columns={"portfolio_net20": "baseline_month_net20"}
        )
        monthly = monthly.merge(baseline, on="eval_month", how="left")
        monthly["delta_vs_baseline_net20"] = (
            pd.to_numeric(monthly["portfolio_net20"], errors="coerce")
            - pd.to_numeric(monthly["baseline_month_net20"], errors="coerce")
        )
    return {
        "monthly": monthly,
        "summary": _aggregate_monthly(monthly),
        "catalog": pd.concat(catalog_rows, ignore_index=True) if catalog_rows else pd.DataFrame(),
        "ledger": pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(),
        "skipped": pd.concat(skipped_rows, ignore_index=True) if skipped_rows else pd.DataFrame(),
    }


def _notes(root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# v2.1E Walk-forward Router Stability",
        "",
        "Status: offline audit only. Rules are derived month by month from prior months only.",
        "No router, shadow, paper-live, or real-live rule is promoted by this report.",
        "",
        "## Walk-forward Candidates",
    ]
    candidates = summary[~summary["rule_id"].eq("baseline_B4")].copy()
    if candidates.empty:
        lines.append("- No walk-forward router candidate was generated.")
    else:
        for row in candidates.sort_values("validation_delta_vs_baseline_net20", ascending=False).head(8).itertuples(index=False):
            lines.append(
                f"- {row.rule_id}: validation_delta={row.validation_delta_vs_baseline_net20:.4%}, "
                f"holdout_delta={row.holdout_delta_vs_baseline_net20:.4%}, "
                f"full_delta={row.delta_vs_baseline_net20:.4%}, "
                f"affected={row.router_affected_events}, improved_months={row.months_improved}/{row.months}."
            )
    promotable = candidates[
        candidates["validation_delta_vs_baseline_net20"].gt(0)
        & candidates["holdout_delta_vs_baseline_net20"].ge(0)
        & candidates["delta_vs_baseline_net20"].gt(0)
    ]
    lines.extend(["", "## Decision"])
    if promotable.empty:
        lines.append("- No prior-only walk-forward router passed validation + holdout stability. Do not add a router shadow.")
    else:
        for row in promotable.sort_values("delta_vs_baseline_net20", ascending=False).head(5).itertuples(index=False):
            lines.append(
                f"- Candidate for further offline review: {row.rule_id}, "
                f"validation_delta={row.validation_delta_vs_baseline_net20:.4%}, "
                f"holdout_delta={row.holdout_delta_vs_baseline_net20:.4%}."
            )
    lines.extend(
        [
            "",
            "Promotion requirement: positive validation delta, non-negative holdout delta, positive full walk-forward delta, and no holdout-derived targets.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21e_walkforward_router_stability(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V21EConfig = V21EConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v21b.v20)
    membership = _prepare_membership(sample, cfg.v21b)
    results = evaluate_walkforward_router(membership, cfg)

    outputs = {
        "walkforward_router_monthly": root / "walkforward_router_monthly.csv",
        "walkforward_router_summary": root / "walkforward_router_summary.csv",
        "walkforward_rule_catalog_by_month": root / "walkforward_rule_catalog_by_month.csv",
        "walkforward_rule_ledger": root / "walkforward_rule_ledger.csv",
        "walkforward_skipped_ledger": root / "walkforward_skipped_ledger.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    results["monthly"].to_csv(outputs["walkforward_router_monthly"], index=False)
    results["summary"].to_csv(outputs["walkforward_router_summary"], index=False)
    results["catalog"].to_csv(outputs["walkforward_rule_catalog_by_month"], index=False)
    results["ledger"].to_csv(outputs["walkforward_rule_ledger"], index=False)
    results["skipped"].to_csv(outputs["walkforward_skipped_ledger"], index=False)
    _notes(root, results["summary"])
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V21EConfig",
    "evaluate_walkforward_router",
    "write_v21e_walkforward_router_stability",
]
