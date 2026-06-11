from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


SOURCE_REPORT_ROOT = Path("reports/v0_9b_portfolio_ranking")
REPORT_ROOT = Path("reports/v0_9b2_ranking_failure_attribution")
FOCAL_COST = 20
FOCAL_POOLS = ["P0_CIC1_ONLY", "P2_CIC1_CIC2_COMBINED"]
KEY_RANKINGS = [
    "first_come_first_served",
    "random",
    "cluster_impulse_density_high",
    "beta_extreme_strength_high",
    "local_volume_shock_strength_high",
    "market_impulse_density_high",
    "liquidity_high",
    "reclaim_quality_high",
    "composite_simple",
    "reverse_beta_extreme_strength",
]
CAPACITY_GRID = ["1", "2", "3", "4", "5", "8", "10", "15", "unlimited"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _max_pos_str(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if len(vals.dropna()) else np.nan


def _status_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    focus = summary[pd.to_numeric(summary.get("cost_single_side_bps"), errors="coerce").eq(FOCAL_COST)].copy()
    focus["max_positions"] = focus["max_positions"].map(_max_pos_str)
    focus["selected_minus_skipped"] = (
        pd.to_numeric(focus.get("net_expectancy"), errors="coerce")
        - pd.to_numeric(focus.get("skipped_avg_net"), errors="coerce")
    )
    focus["lift_vs_random_median"] = (
        pd.to_numeric(focus.get("net_expectancy"), errors="coerce")
        - pd.to_numeric(focus.get("random_median_net"), errors="coerce")
    )
    focus["above_random_p75"] = (
        pd.to_numeric(focus.get("net_expectancy"), errors="coerce")
        > pd.to_numeric(focus.get("random_p75_net"), errors="coerce")
    )
    focus["above_random_p90"] = (
        pd.to_numeric(focus.get("net_expectancy"), errors="coerce")
        > pd.to_numeric(focus.get("random_p90_net"), errors="coerce")
    )
    return focus


def _skip_reason_attribution(source_root: Path) -> pd.DataFrame:
    detail_path = source_root / "portfolio_skipped_timeline.csv"
    if detail_path.exists():
        skipped = _read_csv(detail_path)
        if not skipped.empty:
            rows = []
            for keys, group in skipped.groupby(["pool", "ranking", "max_positions", "skip_reason"], sort=False, dropna=False):
                exits = group.get("exit_reason", pd.Series(dtype=str)).astype(str)
                rows.append(
                    {
                        "source": "skipped_timeline",
                        "candidate_pool": keys[0],
                        "ranking": keys[1],
                        "max_positions": _max_pos_str(keys[2]),
                        "skip_reason": keys[3],
                        "trades": int(len(group)),
                        "net10": _safe_mean(group.get("net_return_10bp", pd.Series(dtype=float))),
                        "net20": _safe_mean(group.get("net_return_20bp", group.get("net_return", pd.Series(dtype=float)))),
                        "avg_holding_minutes": _safe_mean(group.get("holding_minutes", pd.Series(dtype=float))),
                        "tp_rate": float(exits.str.startswith("tp").mean()) if len(exits) else np.nan,
                        "sl_rate": float(exits.str.startswith("sl").mean()) if len(exits) else np.nan,
                        "timeout_rate": float(exits.str.contains("max_hold|open", regex=True).mean()) if len(exits) else np.nan,
                    }
                )
            return pd.DataFrame(rows)
    agg = _read_csv(source_root / "skipped_trade_attribution.csv")
    if agg.empty:
        return pd.DataFrame()
    out = agg.rename(
        columns={
            "pool": "candidate_pool",
            "skipped_trades": "trades",
            "skipped_avg_net20": "net20",
        }
    ).copy()
    out["source"] = "aggregate_skipped_attribution"
    out["net10"] = np.nan
    out["avg_holding_minutes"] = np.nan
    out["tp_rate"] = np.nan
    out["sl_rate"] = np.nan
    out["timeout_rate"] = np.nan
    cols = [
        "source",
        "candidate_pool",
        "ranking",
        "max_positions",
        "skip_reason",
        "trades",
        "net10",
        "net20",
        "avg_holding_minutes",
        "tp_rate",
        "sl_rate",
        "timeout_rate",
    ]
    return out[[col for col in cols if col in out.columns]]


def _conflict_set_proxy(summary: pd.DataFrame) -> pd.DataFrame:
    focus = _status_from_summary(summary)
    if focus.empty:
        return pd.DataFrame()
    rows = []
    focus = focus[
        focus["pool"].astype(str).isin(FOCAL_POOLS)
        & focus["ranking"].astype(str).isin(KEY_RANKINGS)
        & focus["max_positions"].astype(str).isin(["1", "3", "5", "10"])
    ].copy()
    for row in focus.itertuples(index=False):
        selected = int(getattr(row, "selected_trades", 0))
        skipped = int(getattr(row, "skipped_trades", 0))
        rows.append(
            {
                "conflict_set_id": f"aggregate::{row.pool}::{row.ranking}::max{row.max_positions}",
                "analysis_level": "aggregate_proxy",
                "candidate_pool": row.pool,
                "ranking": row.ranking,
                "max_positions": row.max_positions,
                "available_slots": row.max_positions,
                "num_candidates": selected + skipped,
                "selected_count": selected,
                "skipped_count": skipped,
                "selected_net20": getattr(row, "net_expectancy", np.nan),
                "skipped_net20": getattr(row, "skipped_avg_net", np.nan),
                "oracle_topk_net20": np.nan,
                "random_topk_mean_net20": getattr(row, "random_median_net", np.nan),
                "ranker_regret_vs_oracle": np.nan,
                "ranker_lift_vs_random": getattr(row, "lift_vs_random_median", np.nan),
                "selected_minus_skipped": getattr(row, "selected_minus_skipped", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _feature_bucket_report(source_root: Path) -> pd.DataFrame:
    buckets = _read_csv(source_root / "feature_rank_bucket_summary.csv")
    if buckets.empty:
        return pd.DataFrame()
    out = buckets.copy()
    out["cost_single_side_bps"] = FOCAL_COST
    out["bucket_order"] = out["bucket"].astype(str).str.extract(r"q(\d+)").astype(float)
    out["month_cap35_net20"] = np.nan
    return out.rename(columns={"pool": "candidate_pool"})


def _cluster_density_shape(feature_buckets: pd.DataFrame) -> pd.DataFrame:
    if feature_buckets.empty:
        return pd.DataFrame()
    sample = feature_buckets[feature_buckets["feature"].astype(str).eq("cluster_impulse_density")].copy()
    if sample.empty:
        return pd.DataFrame()
    rows = []
    for pool, group in sample.groupby("candidate_pool", sort=False, dropna=False):
        ordered = group.sort_values("bucket_order")
        best = ordered.loc[pd.to_numeric(ordered["net20"], errors="coerce").idxmax()]
        high = ordered[ordered["bucket"].astype(str).str.contains("high", case=False, na=False)]
        high_net = _safe_mean(high["net20"]) if not high.empty else np.nan
        rows.append(
            {
                "candidate_pool": pool,
                "feature": "cluster_impulse_density",
                "best_bucket": best["bucket"],
                "best_bucket_net20": float(best["net20"]),
                "highest_bucket_net20": high_net,
                "best_minus_highest_bucket": float(best["net20"] - high_net) if pd.notna(high_net) else np.nan,
                "interpretation": "nonlinear_or_crowding_possible"
                if pd.notna(high_net) and float(best["net20"]) > high_net
                else "high_bucket_not_worse",
            }
        )
    return pd.DataFrame(rows)


def _capacity_curve_extended(source_root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    path = source_root / "capacity_curve.csv"
    if path.exists():
        base = _read_csv(path)
    else:
        base = pd.DataFrame()
    if base.empty:
        focus = summary.copy()
        focus["max_positions"] = focus["max_positions"].map(_max_pos_str)
        if "net_expectancy" in focus.columns:
            focus = focus.rename(columns={"net_expectancy": "20bp", "skipped_avg_net": "skipped_20bp"})
        base_cols = [
            "pool",
            "ranking",
            "max_positions",
            "selected_trades",
            "skipped_trades",
            "20bp",
            "skipped_20bp",
        ]
        base = focus[[col for col in base_cols if col in focus.columns]].copy()
        if "20bp" in base.columns and "skipped_20bp" in base.columns:
            base["selected_minus_skipped_20bp"] = base["20bp"] - base["skipped_20bp"]
    if base.empty:
        return pd.DataFrame()
    base = base.copy()
    base["max_positions"] = base["max_positions"].map(_max_pos_str)
    rows = []
    keys = base[["pool", "ranking"]].drop_duplicates()
    for item in keys.itertuples(index=False):
        local = base[base["pool"].eq(item.pool) & base["ranking"].eq(item.ranking)].copy()
        for pos in CAPACITY_GRID:
            found = local[local["max_positions"].eq(pos)]
            if found.empty:
                rows.append({"pool": item.pool, "ranking": item.ranking, "max_positions": pos, "available": False})
            else:
                row = found.iloc[0].to_dict()
                row["available"] = True
                rows.append(row)
    return pd.DataFrame(rows)


def _ranking_rule_sanity(summary: pd.DataFrame) -> pd.DataFrame:
    focus = _status_from_summary(summary)
    if focus.empty:
        return pd.DataFrame()
    focus = focus[
        focus["pool"].astype(str).isin(FOCAL_POOLS)
        & focus["ranking"].astype(str).isin(KEY_RANKINGS)
        & focus["max_positions"].astype(str).isin(["3", "5", "10"])
    ].copy()
    focus["selected_gt_skipped"] = focus["selected_minus_skipped"] > 0
    focus["net20_positive"] = pd.to_numeric(focus["net_expectancy"], errors="coerce") > 0
    focus["passes_shadow_ranking_bar"] = (
        focus["net20_positive"] & focus["selected_gt_skipped"] & focus["above_random_p75"]
    )
    cols = [
        "pool",
        "ranking",
        "max_positions",
        "selected_trades",
        "skipped_trades",
        "net_expectancy",
        "skipped_avg_net",
        "selected_minus_skipped",
        "random_median_net",
        "random_p75_net",
        "random_p90_net",
        "percentile_vs_random",
        "above_random_p75",
        "above_random_p90",
        "selected_gt_skipped",
        "net20_positive",
        "month_cap35_net",
        "passes_shadow_ranking_bar",
    ]
    return focus[[col for col in cols if col in focus.columns]]


def _signal_burst_order(source_root: Path) -> pd.DataFrame:
    timeline = _read_csv(source_root / "portfolio_timeline.csv")
    if timeline.empty:
        return pd.DataFrame()
    data = timeline.copy()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    data = data[pd.to_numeric(data.get("cost_single_side_bps"), errors="coerce").eq(FOCAL_COST)].copy()
    if data.empty:
        return pd.DataFrame()
    data["burst_window_1h"] = data["entry_time"].dt.floor("1h")
    data["burst_order"] = (
        data.sort_values(["entry_time", "symbol"])
        .groupby(["pool", "ranking", "max_positions", "burst_window_1h"], sort=False)
        .cumcount()
        + 1
    )
    data["burst_size"] = data.groupby(["pool", "ranking", "max_positions", "burst_window_1h"])["symbol"].transform("size")
    rows = []
    for keys, group in data.groupby(["pool", "ranking", "max_positions", "burst_order"], sort=False, dropna=False):
        net = pd.to_numeric(group.get("net_return"), errors="coerce")
        rows.append(
            {
                "source": "selected_timeline_only",
                "candidate_pool": keys[0],
                "ranking": keys[1],
                "max_positions": _max_pos_str(keys[2]),
                "burst_order": int(keys[3]),
                "trades": int(len(group)),
                "avg_burst_size": _safe_mean(group["burst_size"]),
                "net20": _safe_mean(net),
                "good_trade_rate": float((net > 0).mean()) if len(net) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    report_root: Path,
    conflict: pd.DataFrame,
    cluster_shape: pd.DataFrame,
    ranking: pd.DataFrame,
    capacity: pd.DataFrame,
) -> None:
    lines = [
        "# v0.9B.2 Ranking Failure Attribution",
        "",
        "Purpose: explain why max_positions-constrained selected trades are weaker than skipped counterfactuals.",
        "",
        "No CIC/MIR1 parameters were changed. This report is attribution only.",
        "",
    ]
    if not conflict.empty:
        bad = conflict[
            pd.to_numeric(conflict["selected_minus_skipped"], errors="coerce") < 0
        ].sort_values("selected_minus_skipped").head(5)
        lines.append("## Selected vs Skipped")
        for row in bad.itertuples(index=False):
            lines.append(
                f"- {row.candidate_pool}/{row.ranking}/max{row.max_positions}: "
                f"selected_net20={row.selected_net20:.4%}, skipped_net20={row.skipped_net20:.4%}, "
                f"delta={row.selected_minus_skipped:.4%}."
            )
    if not cluster_shape.empty:
        lines.extend(["", "## Cluster Density Shape"])
        for row in cluster_shape.itertuples(index=False):
            lines.append(
                f"- {row.candidate_pool}: best_bucket={row.best_bucket}, "
                f"best_net20={row.best_bucket_net20:.4%}, highest_bucket_net20={row.highest_bucket_net20:.4%}."
            )
    if not ranking.empty:
        passed = ranking[ranking["passes_shadow_ranking_bar"].fillna(False)]
        lines.extend(["", "## Ranking Rule Sanity"])
        lines.append(f"- rules_passing_shadow_bar: {len(passed)}")
        if passed.empty:
            lines.append("- No ranking rule both beats skipped and random p75 under the current summary inputs.")
    if not capacity.empty:
        unavailable = capacity[~capacity["available"].fillna(False)]
        lines.extend(["", "## Capacity Grid"])
        lines.append(f"- unavailable_capacity_points: {len(unavailable)}")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v09b2_ranking_failure_attribution(
    source_root: Path = SOURCE_REPORT_ROOT,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    summary = _read_csv(source_root / "ranking_summary.csv")
    skip_attr = _skip_reason_attribution(source_root)
    conflict = _conflict_set_proxy(summary)
    feature_buckets = _feature_bucket_report(source_root)
    cluster_shape = _cluster_density_shape(feature_buckets)
    capacity = _capacity_curve_extended(source_root, summary)
    ranking = _ranking_rule_sanity(summary)
    burst = _signal_burst_order(source_root)
    outputs = {
        "skip_reason_attribution": report_root / "skip_reason_attribution.csv",
        "conflict_set_analysis": report_root / "conflict_set_analysis.csv",
        "ranking_feature_bucket_summary": report_root / "ranking_feature_bucket_summary.csv",
        "cluster_density_shape": report_root / "cluster_density_shape.csv",
        "capacity_curve_extended": report_root / "capacity_curve_extended.csv",
        "ranking_rule_sanity": report_root / "ranking_rule_sanity.csv",
        "signal_burst_order": report_root / "signal_burst_order.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    skip_attr.to_csv(outputs["skip_reason_attribution"], index=False)
    conflict.to_csv(outputs["conflict_set_analysis"], index=False)
    feature_buckets.to_csv(outputs["ranking_feature_bucket_summary"], index=False)
    cluster_shape.to_csv(outputs["cluster_density_shape"], index=False)
    capacity.to_csv(outputs["capacity_curve_extended"], index=False)
    ranking.to_csv(outputs["ranking_rule_sanity"], index=False)
    burst.to_csv(outputs["signal_burst_order"], index=False)
    _write_notes(report_root, conflict, cluster_shape, ranking, capacity)
    return outputs


__all__ = ["REPORT_ROOT", "write_v09b2_ranking_failure_attribution"]
