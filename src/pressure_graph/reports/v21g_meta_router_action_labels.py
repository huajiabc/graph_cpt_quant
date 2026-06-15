"""v2.1G Meta-router action-label dataset.

Builds an offline, as-of feature + counterfactual label table for later
meta-router research. This report does not fit a model and does not create a
live gate, selector, or management rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import V20Config, _prepare_sample
from pressure_graph.reports.v21b_state_cluster_atlas import V21BConfig, _prepare_membership
from pressure_graph.reports.v21f_state_drift_audit import DRIFT_FEATURES, _available_features, _walkforward_novelty


REPORT_ROOT = Path("reports/v2_1g_meta_router_action_labels")
STATE_COLUMNS = [
    "state_cluster_id",
    "cic_type",
    "btc_state",
    "market_impulse_density",
    "cluster_density",
    "beta_strength",
    "local_shock_strength",
    "ret_4h",
    "ret_4h_percentile",
    "symbol_volatility_percentile",
    "liquidity_rank",
    "burst_count_so_far",
    "minutes_since_burst_start",
    "same_timestamp_peer_count",
    "walkforward_state_novelty",
    "novelty_bucket",
]


@dataclass(frozen=True)
class V21GConfig:
    report_root: Path = REPORT_ROOT
    v21b: V21BConfig = V21BConfig(v20=V20Config())
    label_margin: float = 0.001
    small_trade_margin: float = 0.003


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _entry_month(frame: pd.DataFrame) -> pd.Series:
    entry = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    return entry.dt.strftime("%Y-%m")


def _ensure_trade_key(frame: pd.DataFrame) -> pd.Series:
    if "trade_key" in frame.columns:
        return frame["trade_key"].astype(str)
    if "signal_id" in frame.columns:
        return frame["signal_id"].astype(str)
    return (
        frame.get("symbol", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + pd.to_datetime(frame["entry_time"], utc=True, errors="coerce").astype(str)
        + "|"
        + frame.get("candidate", pd.Series("", index=frame.index)).astype(str)
    )


def _pre_entry_label(net20: pd.Series, cfg: V21GConfig) -> pd.Series:
    out = pd.Series("small_or_neutral", index=net20.index, dtype="object")
    out.loc[net20.gt(cfg.small_trade_margin)] = "core_trade"
    out.loc[net20.lt(-cfg.label_margin)] = "no_trade"
    return out


def _checkpoint_label(frame: pd.DataFrame, cfg: V21GConfig) -> pd.Series:
    out = pd.Series("not_cp60_eligible", index=frame.index, dtype="object")
    eligible = frame.get("cp60_would_exit", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    delta_exit_vs_keep = _num(frame, "checkpoint_net_at_cost") - _num(frame, "net20")
    out.loc[eligible & delta_exit_vs_keep.gt(cfg.label_margin)] = "cp60_exit_better"
    out.loc[eligible & delta_exit_vs_keep.lt(-cfg.label_margin)] = "keep_better_false_exit_risk"
    out.loc[eligible & delta_exit_vs_keep.abs().le(cfg.label_margin)] = "checkpoint_neutral"
    return out


def _protect_label(frame: pd.DataFrame, cfg: V21GConfig) -> pd.Series:
    out = pd.Series("not_protect_eligible", index=frame.index, dtype="object")
    eligible = frame.get("beta_high_protect_candidate", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    delta_keep_vs_cp60 = _num(frame, "net20") - _num(frame, "checkpoint_net_at_cost")
    out.loc[eligible & delta_keep_vs_cp60.gt(cfg.label_margin)] = "protect_keep_better"
    out.loc[eligible & delta_keep_vs_cp60.lt(-cfg.label_margin)] = "cp60_exit_better"
    out.loc[eligible & delta_keep_vs_cp60.abs().le(cfg.label_margin)] = "protect_neutral"
    return out


def _capacity_label(frame: pd.DataFrame, cfg: V21GConfig) -> pd.Series:
    out = pd.Series("not_overflow_eligible", index=frame.index, dtype="object")
    eligible = frame.get("late_burst_o6_candidate", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    net20 = _num(frame, "net20")
    out.loc[eligible & net20.gt(cfg.small_trade_margin)] = "allow_overflow"
    out.loc[eligible & net20.lt(-cfg.label_margin)] = "deny_overflow"
    out.loc[eligible & net20.between(-cfg.label_margin, cfg.small_trade_margin, inclusive="both")] = "overflow_neutral"
    return out


def _walkforward_novelty_for_membership(membership: pd.DataFrame, cfg: V21GConfig) -> pd.DataFrame:
    from pressure_graph.reports.v21f_state_drift_audit import V21FConfig

    features = _available_features(membership)
    return _walkforward_novelty(
        membership,
        features,
        V21FConfig(v21b=cfg.v21b),
    )


def build_meta_router_labels(membership: pd.DataFrame, cfg: V21GConfig = V21GConfig()) -> pd.DataFrame:
    data = membership.copy()
    data["trade_key"] = _ensure_trade_key(data)
    data["entry_month"] = _entry_month(data)
    novelty = _walkforward_novelty_for_membership(data, cfg)
    novelty_cols = [
        "trade_key",
        "novelty_status",
        "walkforward_state_novelty",
        "novelty_feature_count",
        "dominant_novelty_features",
        "cluster_reference_events",
        "novelty_bucket",
    ]
    available = [col for col in novelty_cols if col in novelty.columns]
    data = data.merge(novelty[available], on="trade_key", how="left", suffixes=("", "_novelty"))
    data["net20"] = _num(data, "net20")
    data["checkpoint_exit_delta_vs_keep"] = _num(data, "checkpoint_net_at_cost") - data["net20"]
    data["protect_keep_delta_vs_cp60"] = data["net20"] - _num(data, "checkpoint_net_at_cost")
    data["utility_core_trade"] = data["net20"]
    data["utility_no_trade"] = 0.0
    data["utility_reduce_size_50"] = data["net20"] * 0.5
    data["utility_cp60_exit_if_triggered"] = np.where(
        data.get("cp60_would_exit", pd.Series(False, index=data.index)).fillna(False).astype(bool),
        _num(data, "checkpoint_net_at_cost"),
        np.nan,
    )
    data["utility_o6_overflow_025"] = np.where(
        data.get("late_burst_o6_candidate", pd.Series(False, index=data.index)).fillna(False).astype(bool),
        data["net20"] * 0.25,
        np.nan,
    )
    data["utility_o6_overflow_050"] = np.where(
        data.get("late_burst_o6_candidate", pd.Series(False, index=data.index)).fillna(False).astype(bool),
        data["net20"] * 0.50,
        np.nan,
    )
    data["pre_entry_action_label"] = _pre_entry_label(data["net20"], cfg)
    data["post_entry_checkpoint_label"] = _checkpoint_label(data, cfg)
    data["protect_a_label"] = _protect_label(data, cfg)
    data["capacity_overflow_label"] = _capacity_label(data, cfg)
    data["label_margin"] = cfg.label_margin
    data["small_trade_margin"] = cfg.small_trade_margin
    data["label_source"] = "realized_counterfactual_not_asof_feature"
    data["meta_router_training_split"] = data["period"].astype(str)
    return data.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _label_summary(labels: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_cols = ["pre_entry_action_label", "post_entry_checkpoint_label", "protect_a_label", "capacity_overflow_label"]
    for label_col in label_cols:
        for keys, group in labels.groupby([*group_cols, label_col], sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip([*group_cols, label_col], keys, strict=False)}
            net = _num(group, "net20")
            row.update(
                {
                    "label_head": label_col,
                    "events": int(len(group)),
                    "net20_sum": float(net.sum()),
                    "net20_avg": float(net.mean()) if len(group) else np.nan,
                    "hit_rate": float(net.gt(0).mean()) if len(group) else np.nan,
                    "cp60_delta_sum": float(_num(group, "checkpoint_exit_delta_vs_keep").sum()),
                    "protect_delta_sum": float(_num(group, "protect_keep_delta_vs_cp60").sum()),
                    "o6_025_utility_sum": float(_num(group, "utility_o6_overflow_025").sum()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_matrix(labels: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "trade_key",
        "symbol",
        "candidate",
        "entry_time",
        "entry_month",
        "period",
        "meta_router_training_split",
    ]
    action_cols = [
        "net20",
        "pre_entry_action_label",
        "post_entry_checkpoint_label",
        "protect_a_label",
        "capacity_overflow_label",
        "utility_core_trade",
        "utility_no_trade",
        "utility_reduce_size_50",
        "utility_cp60_exit_if_triggered",
        "utility_o6_overflow_025",
        "utility_o6_overflow_050",
        "checkpoint_exit_delta_vs_keep",
        "protect_keep_delta_vs_cp60",
    ]
    cols = [col for col in [*base_cols, *STATE_COLUMNS, *action_cols] if col in labels.columns]
    return labels[cols].copy()


def _schema(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in labels.columns:
        role = "feature" if col in STATE_COLUMNS or col in DRIFT_FEATURES else "metadata"
        if col.endswith("_label"):
            role = "label"
        if col.startswith("utility_") or col.endswith("_delta_vs_keep") or col.endswith("_delta_vs_cp60"):
            role = "counterfactual_utility"
        rows.append(
            {
                "column": col,
                "dtype": str(labels[col].dtype),
                "role": role,
                "non_null": int(labels[col].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _notes(root: Path, labels: pd.DataFrame, period_summary: pd.DataFrame, state_summary: pd.DataFrame) -> None:
    lines = [
        "# v2.1G Meta-router Action Labels",
        "",
        "Status: offline dataset only. No model, router, gate, paper-live, or real-live rule is promoted.",
        "",
        f"Events: {len(labels)}",
        "",
        "## Label Heads",
    ]
    for label_col in ["pre_entry_action_label", "post_entry_checkpoint_label", "protect_a_label", "capacity_overflow_label"]:
        counts = labels[label_col].value_counts(dropna=False)
        parts = [f"{idx}={count}" for idx, count in counts.items()]
        lines.append(f"- {label_col}: " + ", ".join(parts))
    lines.extend(["", "## Period Label Snapshot"])
    pre = period_summary[period_summary["label_head"].eq("pre_entry_action_label")].copy()
    if not pre.empty:
        for row in pre.sort_values(["period", "pre_entry_action_label"]).itertuples(index=False):
            lines.append(
                f"- {row.period}/{row.pre_entry_action_label}: events={row.events}, "
                f"net20_sum={row.net20_sum:.4%}, avg={row.net20_avg:.4%}."
            )
    lines.extend(["", "## Weak State Label Concentration"])
    weak = state_summary[
        state_summary["label_head"].eq("pre_entry_action_label")
        & state_summary["pre_entry_action_label"].eq("no_trade")
    ].copy()
    if weak.empty:
        lines.append("- No no-trade labels found by state.")
    else:
        for row in weak.sort_values("net20_sum", ascending=True).head(6).itertuples(index=False):
            lines.append(
                f"- {row.state_cluster_id}: no_trade_events={row.events}, "
                f"net20_sum={row.net20_sum:.4%}, avg={row.net20_avg:.4%}."
            )
    lines.extend(
        [
            "",
            "Decision rule: this table is the input dataset for v2.2 router research. It is not itself a selector.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21g_meta_router_action_labels(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V21GConfig = V21GConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v21b.v20)
    membership = _prepare_membership(sample, cfg.v21b)
    labels = build_meta_router_labels(membership, cfg)
    feature_matrix = _feature_matrix(labels)
    period_summary = _label_summary(labels, ["period"])
    state_summary = _label_summary(labels, ["state_cluster_id"])
    schema = _schema(feature_matrix)

    outputs = {
        "meta_router_event_labels": root / "meta_router_event_labels.csv",
        "meta_router_feature_matrix": root / "meta_router_feature_matrix.csv",
        "period_action_label_summary": root / "period_action_label_summary.csv",
        "state_action_label_summary": root / "state_action_label_summary.csv",
        "router_dataset_schema": root / "router_dataset_schema.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    labels.to_csv(outputs["meta_router_event_labels"], index=False)
    feature_matrix.to_csv(outputs["meta_router_feature_matrix"], index=False)
    period_summary.to_csv(outputs["period_action_label_summary"], index=False)
    state_summary.to_csv(outputs["state_action_label_summary"], index=False)
    schema.to_csv(outputs["router_dataset_schema"], index=False)
    _notes(root, labels, period_summary, state_summary)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "STATE_COLUMNS",
    "V21GConfig",
    "build_meta_router_labels",
    "write_v21g_meta_router_action_labels",
]
