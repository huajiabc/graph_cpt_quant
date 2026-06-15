"""v2.1D State Router Preflight.

Tests simple no-trade / reduce-size actions over discovered state clusters and
state-transition keys. The report separates pre-holdout candidates from
holdout-derived diagnostics to avoid accidental promotion of leakage rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import (
    MotifSpec,
    V20Config,
    _prepare_sample,
    _simulate_portfolio,
)
from pressure_graph.reports.v21b_state_cluster_atlas import V21BConfig, _prepare_membership


REPORT_ROOT = Path("reports/v2_1d_router_preflight")
BASE_SPEC = MotifSpec(
    max_positions=8,
    overflow_rule="O6_late9",
    checkpoint_rule="Protect_A_cap2",
    protect_cap=2,
)


@dataclass(frozen=True)
class V21DConfig:
    report_root: Path = REPORT_ROOT
    v21b: V21BConfig = V21BConfig(v20=V20Config())
    reduce_size_multiplier: float = 0.5
    min_cluster_events: int = 5
    min_transition_events: int = 3


def _transition_key(relation: str, source: pd.Series, target: pd.Series) -> pd.Series:
    return relation + "|h1|" + source.fillna("none").astype(str) + "->" + target.fillna("none").astype(str)


def _add_transition_keys(membership: pd.DataFrame) -> pd.DataFrame:
    out = membership.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    ordered = out.sort_values(["entry_time", "symbol", "candidate"]).copy()
    ordered["prev_global_cluster"] = ordered["state_cluster_id"].shift(1)
    ordered["global_h1_key"] = _transition_key(
        "global_time_event",
        ordered["prev_global_cluster"],
        ordered["state_cluster_id"],
    )
    ordered["prev_symbol_cluster"] = ordered.groupby("symbol", sort=False)["state_cluster_id"].shift(1)
    ordered["same_symbol_h1_key"] = _transition_key(
        "same_symbol_event",
        ordered["prev_symbol_cluster"],
        ordered["state_cluster_id"],
    )
    ordered["prev_burst_cluster"] = ordered.groupby("burst_id", sort=False, dropna=False)["state_cluster_id"].shift(1)
    ordered["same_burst_h1_key"] = _transition_key(
        "same_burst_event",
        ordered["prev_burst_cluster"],
        ordered["state_cluster_id"],
    )
    for col in ("global_h1_key", "same_symbol_h1_key", "same_burst_h1_key"):
        ordered.loc[ordered[col].str.contains("none->", regex=False, na=False), col] = ""
    return ordered.sort_index()


def _cluster_stats(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in membership.groupby("state_cluster_id", sort=False):
        periods = group.groupby("period", sort=False)["net20"].agg(["count", "sum", "mean"]).reset_index()
        row: dict[str, Any] = {"state_cluster_id": cluster_id}
        for period in ("search", "validation", "holdout"):
            part = periods[periods["period"].eq(period)]
            row[f"{period}_events"] = int(part["count"].iloc[0]) if not part.empty else 0
            row[f"{period}_net20"] = float(part["sum"].iloc[0]) if not part.empty else 0.0
            row[f"{period}_avg_net20"] = float(part["mean"].iloc[0]) if not part.empty else np.nan
        row["preholdout_events"] = int(row["search_events"] + row["validation_events"])
        row["preholdout_net20"] = float(row["search_net20"] + row["validation_net20"])
        rows.append(row)
    return pd.DataFrame(rows)


def _transition_stats(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_cols = ["global_h1_key", "same_symbol_h1_key", "same_burst_h1_key"]
    for col in key_cols:
        relation = col.replace("_h1_key", "")
        valid = membership[membership[col].astype(str).ne("")].copy()
        for key, group in valid.groupby(col, sort=False):
            periods = group.groupby("period", sort=False)["net20"].agg(["count", "sum", "mean"]).reset_index()
            row: dict[str, Any] = {"relation": relation, "transition_key": key}
            for period in ("search", "validation", "holdout"):
                part = periods[periods["period"].eq(period)]
                row[f"{period}_events"] = int(part["count"].iloc[0]) if not part.empty else 0
                row[f"{period}_net20"] = float(part["sum"].iloc[0]) if not part.empty else 0.0
                row[f"{period}_avg_net20"] = float(part["mean"].iloc[0]) if not part.empty else np.nan
            row["preholdout_events"] = int(row["search_events"] + row["validation_events"])
            row["preholdout_net20"] = float(row["search_net20"] + row["validation_net20"])
            rows.append(row)
    return pd.DataFrame(rows)


def _build_rules(cluster_stats: pd.DataFrame, transition_stats: pd.DataFrame, cfg: V21DConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "rule_id": "baseline_B4",
            "rule_type": "baseline",
            "action": "none",
            "source": "benchmark",
            "target_values": "",
            "leakage_status": "safe",
        }
    ]
    pre_bad_clusters = cluster_stats[
        cluster_stats["preholdout_events"].ge(cfg.min_cluster_events)
        & cluster_stats["preholdout_net20"].lt(0)
    ]["state_cluster_id"].astype(str).tolist()
    flip_clusters = cluster_stats[
        cluster_stats["preholdout_net20"].gt(0)
        & cluster_stats["holdout_net20"].lt(0)
        & cluster_stats["holdout_events"].gt(0)
    ]["state_cluster_id"].astype(str).tolist()
    pre_bad_transitions = transition_stats[
        transition_stats["preholdout_events"].ge(cfg.min_transition_events)
        & transition_stats["preholdout_net20"].lt(0)
    ]["transition_key"].astype(str).tolist()
    holdout_weak_transitions = transition_stats[
        transition_stats["holdout_events"].ge(2)
        & transition_stats["holdout_net20"].lt(0)
    ]["transition_key"].astype(str).tolist()

    candidates = [
        ("preholdout_bad_clusters", "cluster", pre_bad_clusters, "safe"),
        ("holdout_flip_clusters_diagnostic", "cluster", flip_clusters, "holdout_leakage_diagnostic"),
        ("preholdout_bad_transitions", "transition", pre_bad_transitions, "safe"),
        ("holdout_weak_transitions_diagnostic", "transition", holdout_weak_transitions, "holdout_leakage_diagnostic"),
    ]
    for source, rule_type, values, leakage in candidates:
        values = sorted(set(value for value in values if value))
        if not values:
            continue
        for action in ("no_trade", "reduce_size_50"):
            rows.append(
                {
                    "rule_id": f"{action}_{source}",
                    "rule_type": rule_type,
                    "action": action,
                    "source": source,
                    "target_values": "|".join(values),
                    "leakage_status": leakage,
                }
            )
    return pd.DataFrame(rows)


def _rule_mask(sample: pd.DataFrame, rule: pd.Series) -> pd.Series:
    values = [value for value in str(rule.get("target_values", "")).split("|") if value]
    if not values or rule.get("rule_type") == "baseline":
        return pd.Series(False, index=sample.index)
    if rule.get("rule_type") == "cluster":
        return sample["state_cluster_id"].astype(str).isin(values)
    if rule.get("rule_type") == "transition":
        masks = [
            sample.get(col, pd.Series("", index=sample.index)).astype(str).isin(values)
            for col in ("global_h1_key", "same_symbol_h1_key", "same_burst_h1_key")
        ]
        return masks[0] | masks[1] | masks[2]
    return pd.Series(False, index=sample.index)


def _ledger_metrics(ledger: pd.DataFrame, denominator: int) -> dict[str, Any]:
    if ledger.empty:
        return {
            "selected_trades": 0,
            "portfolio_net20": 0.0,
            "month_cap35_net20": np.nan,
            "worst_month_net20": np.nan,
            "avg_trade_net20": np.nan,
            "hit_rate": np.nan,
        }
    weighted = pd.to_numeric(ledger["weighted_return"], errors="coerce")
    contribution = weighted / max(1, denominator)
    months = contribution.groupby(ledger["month"].astype(str), sort=False).sum()
    total = float(contribution.sum())
    cap_value = total * 0.35 if total > 0 else 0.0
    capped = [min(value, cap_value) if value > 0 and cap_value > 0 else value for value in months]
    trade_net = pd.to_numeric(ledger.get("effective_net_return"), errors="coerce")
    return {
        "selected_trades": int(len(ledger)),
        "portfolio_net20": total,
        "month_cap35_net20": float(np.sum(capped)) if len(capped) else np.nan,
        "worst_month_net20": float(months.min()) if len(months) else np.nan,
        "avg_trade_net20": float(trade_net.mean()) if len(trade_net) else np.nan,
        "hit_rate": float(trade_net.gt(0).mean()) if len(trade_net) else np.nan,
    }


def _simulate_rule(sample: pd.DataFrame, rule: pd.Series, cfg: V21DConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = _rule_mask(sample, rule)
    action = str(rule.get("action"))
    router_skipped = 0
    if action == "no_trade":
        router_skipped = int(mask.sum())
        sim_sample = sample[~mask].copy()
        ledger, skipped = _simulate_portfolio(sim_sample, BASE_SPEC)
    else:
        ledger, skipped = _simulate_portfolio(sample, BASE_SPEC)
        if action == "reduce_size_50" and not ledger.empty:
            selected_mask = _rule_mask(ledger, rule)
            ledger = ledger.copy()
            ledger["router_reduced_size"] = selected_mask
            ledger.loc[selected_mask, "weighted_return"] = (
                pd.to_numeric(ledger.loc[selected_mask, "weighted_return"], errors="coerce")
                * float(cfg.reduce_size_multiplier)
            )
            router_skipped = int(selected_mask.sum())
    metrics = _ledger_metrics(ledger, BASE_SPEC.max_positions)
    metrics.update(
        {
            "router_affected_events": router_skipped,
            "skipped_trades": int(len(skipped)),
        }
    )
    return ledger, metrics


def _evaluate_rules(membership: pd.DataFrame, rules: pd.DataFrame, cfg: V21DConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for _, rule in rules.iterrows():
        row = rule.to_dict()
        for period in ("search", "validation", "holdout", "full"):
            period_sample = membership if period == "full" else membership[membership["period"].eq(period)].copy()
            ledger, metrics = _simulate_rule(period_sample, rule, cfg)
            if not ledger.empty:
                ledger = ledger.copy()
                ledger["rule_id"] = rule["rule_id"]
                ledger["period_eval"] = period
                ledgers.append(ledger)
            for key, value in metrics.items():
                row[f"{period}_{key}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary[summary["rule_id"].eq("baseline_B4")]
    if not baseline.empty:
        base = baseline.iloc[0]
        for period in ("search", "validation", "holdout", "full"):
            summary[f"{period}_delta_vs_baseline_net20"] = (
                pd.to_numeric(summary[f"{period}_portfolio_net20"], errors="coerce")
                - float(base[f"{period}_portfolio_net20"])
            )
    ledger_out = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    return summary.sort_values(["leakage_status", "holdout_delta_vs_baseline_net20"], ascending=[True, False]), ledger_out


def _notes(root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# v2.1D Router Preflight",
        "",
        "Status: offline preflight only. No router, paper-live, or real-live rule changes.",
        "",
        "## Safe Candidates",
    ]
    safe = summary[summary["leakage_status"].eq("safe") & ~summary["rule_id"].eq("baseline_B4")].copy()
    if safe.empty:
        lines.append("- No safe pre-holdout router candidate was generated.")
    else:
        for row in safe.sort_values("validation_delta_vs_baseline_net20", ascending=False).head(8).itertuples(index=False):
            lines.append(
                f"- {row.rule_id}: validation_delta={row.validation_delta_vs_baseline_net20:.4%}, "
                f"holdout_delta={row.holdout_delta_vs_baseline_net20:.4%}, "
                f"validation_net={row.validation_portfolio_net20:.4%}, holdout_net={row.holdout_portfolio_net20:.4%}."
            )
    diag = summary[summary["leakage_status"].eq("holdout_leakage_diagnostic")].copy()
    if not diag.empty:
        lines.extend(["", "## Holdout-Derived Diagnostics"])
        for row in diag.sort_values("holdout_delta_vs_baseline_net20", ascending=False).head(8).itertuples(index=False):
            lines.append(
                f"- {row.rule_id}: holdout_delta={row.holdout_delta_vs_baseline_net20:.4%}, "
                f"validation_delta={row.validation_delta_vs_baseline_net20:.4%}, "
                "not promotable because it is derived from holdout damage."
            )
    lines.extend(
        [
            "",
            "Decision rule: a router candidate needs positive validation delta and non-worse holdout delta without using holdout-derived targets before it can move to shadow design.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21d_router_preflight(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V21DConfig = V21DConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v21b.v20)
    membership = _add_transition_keys(_prepare_membership(sample, cfg.v21b))
    clusters = _cluster_stats(membership)
    transitions = _transition_stats(membership)
    rules = _build_rules(clusters, transitions, cfg)
    summary, ledger = _evaluate_rules(membership, rules, cfg)

    outputs = {
        "router_cluster_stats": root / "router_cluster_stats.csv",
        "router_transition_stats": root / "router_transition_stats.csv",
        "router_rule_catalog": root / "router_rule_catalog.csv",
        "router_rule_summary": root / "router_rule_summary.csv",
        "router_trade_ledger": root / "router_trade_ledger.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    clusters.to_csv(outputs["router_cluster_stats"], index=False)
    transitions.to_csv(outputs["router_transition_stats"], index=False)
    rules.to_csv(outputs["router_rule_catalog"], index=False)
    summary.to_csv(outputs["router_rule_summary"], index=False)
    ledger.to_csv(outputs["router_trade_ledger"], index=False)
    _notes(root, summary)
    return outputs


__all__ = [
    "BASE_SPEC",
    "REPORT_ROOT",
    "V21DConfig",
    "write_v21d_router_preflight",
]
