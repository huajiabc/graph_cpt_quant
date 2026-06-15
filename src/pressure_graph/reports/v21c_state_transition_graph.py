"""v2.1C State Transition Graph.

Builds transition edges between discovered v2.1B state clusters. This is an
offline graph diagnostic, not a trading rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import _prepare_sample
from pressure_graph.reports.v21b_state_cluster_atlas import V21BConfig, _prepare_membership


REPORT_ROOT = Path("reports/v2_1c_state_transition_graph")
EVENT_HORIZONS = (1, 4, 16)


@dataclass(frozen=True)
class V21CConfig:
    report_root: Path = REPORT_ROOT
    v21b: V21BConfig = V21BConfig()


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _edge_rows_for_group(group: pd.DataFrame, relation: str, horizon: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = group.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)
    if len(ordered) <= horizon:
        return rows
    sources = ordered.iloc[:-horizon].copy()
    targets = ordered.iloc[horizon:].copy().reset_index(drop=True)
    sources = sources.reset_index(drop=True)
    for source, target in zip(sources.to_dict("records"), targets.to_dict("records"), strict=False):
        source_time = pd.Timestamp(source["entry_time"])
        target_time = pd.Timestamp(target["entry_time"])
        rows.append(
            {
                "relation": relation,
                "horizon": int(horizon),
                "source_event_id": source.get("signal_id", source.get("trade_key")),
                "target_event_id": target.get("signal_id", target.get("trade_key")),
                "source_symbol": source.get("symbol"),
                "target_symbol": target.get("symbol"),
                "source_cluster": source.get("state_cluster_id"),
                "target_cluster": target.get("state_cluster_id"),
                "source_period": source.get("period"),
                "target_period": target.get("period"),
                "source_entry_time": source_time,
                "target_entry_time": target_time,
                "minutes_to_target": float((target_time - source_time).total_seconds() / 60.0),
                "same_symbol": source.get("symbol") == target.get("symbol"),
                "same_burst": source.get("burst_id") == target.get("burst_id"),
                "source_cic_type": source.get("cic_type"),
                "target_cic_type": target.get("cic_type"),
                "source_net20": source.get("net20"),
                "target_net20": target.get("net20"),
                "target_mfe_12h": target.get("mfe_12h"),
                "target_mae_12h": target.get("mae_12h"),
                "target_hit10_12h": bool(target.get("hit_10pct_12h", False)),
            }
        )
    return rows


def _build_edges(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = membership.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    for horizon in EVENT_HORIZONS:
        for _, group in data.groupby("symbol", sort=False, dropna=False):
            rows.extend(_edge_rows_for_group(group, "same_symbol_event", horizon))
        for _, group in data.groupby("burst_id", sort=False, dropna=False):
            rows.extend(_edge_rows_for_group(group, "same_burst_event", horizon))
        rows.extend(_edge_rows_for_group(data, "global_time_event", horizon))
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    return edges.sort_values(["relation", "horizon", "source_entry_time", "source_symbol"]).reset_index(drop=True)


def _edge_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["relation", "horizon", "source_cluster", "target_cluster"]
    source_totals = edges.groupby(["relation", "horizon", "source_cluster"], sort=False).size()
    for keys, group in edges.groupby(group_cols, sort=False, dropna=False):
        relation, horizon, source_cluster, target_cluster = keys
        total = int(source_totals.loc[(relation, horizon, source_cluster)])
        target_net = pd.to_numeric(group["target_net20"], errors="coerce")
        source_net = pd.to_numeric(group["source_net20"], errors="coerce")
        holdout = group[group["target_period"].astype(str).eq("holdout")]
        rows.append(
            {
                "relation": relation,
                "horizon": int(horizon),
                "source_cluster": source_cluster,
                "target_cluster": target_cluster,
                "transition_count": int(len(group)),
                "transition_probability_from_source": float(len(group) / total) if total else np.nan,
                "avg_source_net20": float(source_net.mean()) if len(source_net) else np.nan,
                "avg_target_net20": float(target_net.mean()) if len(target_net) else np.nan,
                "sum_target_net20": float(target_net.sum()) if len(target_net) else 0.0,
                "target_hit_rate": float(target_net.gt(0).mean()) if len(target_net) else np.nan,
                "target_hit10_12h": float(group["target_hit10_12h"].mean()) if len(group) else np.nan,
                "median_minutes_to_target": float(pd.to_numeric(group["minutes_to_target"], errors="coerce").median()),
                "holdout_transition_count": int(len(holdout)),
                "holdout_target_net20": float(pd.to_numeric(holdout.get("target_net20"), errors="coerce").sum()) if len(holdout) else 0.0,
                "edge_score": float(len(group) / total * target_net.mean()) if total and len(target_net) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["relation", "horizon", "holdout_target_net20", "sum_target_net20"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _horizon_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in edges.groupby(["relation", "horizon"], sort=False, dropna=False):
        relation, horizon = keys
        target_net = pd.to_numeric(group["target_net20"], errors="coerce")
        holdout = group[group["target_period"].astype(str).eq("holdout")]
        rows.append(
            {
                "relation": relation,
                "horizon": int(horizon),
                "edges": int(len(group)),
                "unique_source_clusters": int(group["source_cluster"].nunique()),
                "unique_target_clusters": int(group["target_cluster"].nunique()),
                "avg_target_net20": float(target_net.mean()) if len(group) else np.nan,
                "sum_target_net20": float(target_net.sum()) if len(group) else 0.0,
                "target_hit_rate": float(target_net.gt(0).mean()) if len(group) else np.nan,
                "holdout_edges": int(len(holdout)),
                "holdout_target_net20": float(pd.to_numeric(holdout.get("target_net20"), errors="coerce").sum()) if len(holdout) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _path_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    one_step = edges[edges["horizon"].eq(1)].copy()
    rows: list[dict[str, Any]] = []
    for relation, group in one_step.groupby("relation", sort=False):
        left = group.rename(columns={"target_cluster": "mid_cluster", "source_cluster": "first_cluster"})
        right = group.rename(columns={"source_cluster": "mid_cluster", "target_cluster": "last_cluster"})
        joined = left.merge(
            right,
            left_on=["relation", "mid_cluster", "target_event_id"],
            right_on=["relation", "mid_cluster", "source_event_id"],
            suffixes=("_first", "_second"),
        )
        if joined.empty:
            continue
        for keys, path_group in joined.groupby(["first_cluster", "mid_cluster", "last_cluster"], sort=False):
            first, mid, last = keys
            last_net = pd.to_numeric(path_group["target_net20_second"], errors="coerce")
            rows.append(
                {
                    "relation": relation,
                    "first_cluster": first,
                    "mid_cluster": mid,
                    "last_cluster": last,
                    "path_count": int(len(path_group)),
                    "avg_last_net20": float(last_net.mean()) if len(last_net) else np.nan,
                    "sum_last_net20": float(last_net.sum()) if len(last_net) else 0.0,
                    "last_hit_rate": float(last_net.gt(0).mean()) if len(last_net) else np.nan,
                    "holdout_path_count": int(path_group["target_period_second"].astype(str).eq("holdout").sum()),
                    "holdout_last_net20": float(
                        pd.to_numeric(
                            path_group.loc[path_group["target_period_second"].astype(str).eq("holdout"), "target_net20_second"],
                            errors="coerce",
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["holdout_last_net20", "sum_last_net20"], ascending=[True, False]).reset_index(drop=True)


def _holdout_autopsy(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    holdout = edges[edges["target_period"].astype(str).eq("holdout")].copy()
    if holdout.empty:
        return holdout
    target_net = pd.to_numeric(holdout["target_net20"], errors="coerce")
    holdout["damage_rank"] = target_net.rank(method="first", ascending=True)
    return holdout.sort_values("target_net20", ascending=True).reset_index(drop=True)


def _notes(root: Path, edge_summary: pd.DataFrame, path_summary: pd.DataFrame, holdout: pd.DataFrame) -> None:
    lines = [
        "# v2.1C State Transition Graph",
        "",
        "Status: offline diagnostic only. Transition edges are not gates or live rules.",
        "",
        "## Worst Holdout Transition Edges",
    ]
    bad = edge_summary[edge_summary.get("holdout_transition_count", pd.Series(0, index=edge_summary.index)).fillna(0).gt(0)].copy()
    for row in bad.sort_values("holdout_target_net20").head(8).itertuples(index=False):
        lines.append(
            f"- {row.relation}/h{row.horizon}: {row.source_cluster}->{row.target_cluster}, "
            f"holdout_edges={row.holdout_transition_count}, holdout_target={row.holdout_target_net20:.4%}, "
            f"avg_target={row.avg_target_net20:.4%}."
        )
    if not path_summary.empty:
        lines.extend(["", "## Weak Holdout 2-Step Paths"])
        weak_paths = path_summary[path_summary["holdout_path_count"].gt(0)].sort_values("holdout_last_net20").head(5)
        for row in weak_paths.itertuples(index=False):
            lines.append(
                f"- {row.relation}: {row.first_cluster}->{row.mid_cluster}->{row.last_cluster}, "
                f"holdout_paths={row.holdout_path_count}, holdout_last={row.holdout_last_net20:.4%}."
            )
    if not holdout.empty:
        lines.extend(["", "## Worst Holdout Target Events"])
        for row in holdout.head(5).itertuples(index=False):
            lines.append(
                f"- {row.target_symbol} {row.target_entry_time}: {row.source_cluster}->{row.target_cluster}, "
                f"target_net20={row.target_net20:.4%}, relation={row.relation}/h{row.horizon}."
            )
    lines.extend(
        [
            "",
            "Next: use repeated weak transitions as candidates for v2.2 router no-trade/reduce-size actions, "
            "but only after checking they survive walk-forward validation.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21c_state_transition_graph(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V21CConfig = V21CConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v21b.v20)
    membership = _prepare_membership(sample, cfg.v21b)
    edges = _build_edges(membership)
    edge_summary = _edge_summary(edges)
    horizon = _horizon_summary(edges)
    paths = _path_summary(edges)
    holdout = _holdout_autopsy(edges)

    outputs = {
        "state_transition_edges": root / "state_transition_edges.csv",
        "state_transition_edge_summary": root / "state_transition_edge_summary.csv",
        "state_transition_horizon_summary": root / "state_transition_horizon_summary.csv",
        "state_transition_path_summary": root / "state_transition_path_summary.csv",
        "holdout_transition_autopsy": root / "holdout_transition_autopsy.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    edges.to_csv(outputs["state_transition_edges"], index=False)
    edge_summary.to_csv(outputs["state_transition_edge_summary"], index=False)
    horizon.to_csv(outputs["state_transition_horizon_summary"], index=False)
    paths.to_csv(outputs["state_transition_path_summary"], index=False)
    holdout.to_csv(outputs["holdout_transition_autopsy"], index=False)
    _notes(root, edge_summary, paths, holdout)
    return outputs


__all__ = [
    "EVENT_HORIZONS",
    "REPORT_ROOT",
    "V21CConfig",
    "write_v21c_state_transition_graph",
]
