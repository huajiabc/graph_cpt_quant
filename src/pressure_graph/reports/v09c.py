from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import select_portfolio


REPORT_ROOT = Path("reports/v0_9c_orderflow_capacity_ranking")
SOURCE_ROOT = Path("reports/v0_8_orderflow_shadow")

POOLS = {
    "P0_CIC1_ONLY": ["CIC1_FILTERED_MIR1"],
    "P2_CIC1_CIC2_COMBINED": ["CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"],
}
MAX_POSITIONS = [5, 8, 10]
FOCUS_COST = 20
MIN_DIAGNOSTIC_TRADES = 50
MIN_DECISION_TRADES = 100

ORDERFLOW_FEATURES = [
    "shock_bar_taker_buy_ratio",
    "shock_bar_cvd_delta_turnover",
    "shock_bar_large_buy_turnover",
    "shock_bar_large_sell_turnover",
    "pullback_window_cvd_delta_turnover",
    "pullback_window_buy_sell_imbalance",
    "reclaim_bar_taker_buy_ratio",
    "reclaim_bar_cvd_delta_turnover",
    "reclaim_bar_buy_sell_imbalance",
    "reclaim_bar_large_buy_count",
    "reclaim_bar_large_sell_count",
    "reclaim_bar_large_buy_turnover",
    "reclaim_bar_large_sell_turnover",
    "post_entry_15m_cvd_delta_turnover",
    "post_entry_1h_cvd_delta_turnover",
]

RANKING_RULES = {
    "R0_first_come": "rank_first_come",
    "R2_reclaim_taker_buy_ratio": "rank_reclaim_taker_buy_ratio",
    "R3_reclaim_cvd_delta": "rank_reclaim_cvd_delta",
    "R4_shock_taker_buy_ratio": "rank_shock_taker_buy_ratio",
    "R5_shock_cvd_delta": "rank_shock_cvd_delta",
    "R6_large_buy_volume": "rank_large_buy_volume",
    "R7_sell_pressure_low": "rank_sell_pressure_low",
    "R8_simple_orderflow_composite": "rank_orderflow_composite",
    "R9_old_cluster_rank": "rank_old_cluster",
}


def _read_source(source_root: Path) -> pd.DataFrame:
    path = source_root / "orderflow_shadow_trades.csv"
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _prepare(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    out = data.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    signal_col = "market_gate_time" if "market_gate_time" in out.columns else "entry_time"
    out["signal_time"] = pd.to_datetime(out.get(signal_col), utc=True, errors="coerce")
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    out["cost_single_side_bps"] = FOCUS_COST
    out["net_return"] = _num(out, f"net_return_{FOCUS_COST}bp")
    if "holding_minutes" not in out.columns:
        out["holding_minutes"] = (out["exit_time"] - out["entry_time"]).dt.total_seconds() / 60.0
    out["symbol"] = out["symbol"].astype(str)
    out["candidate"] = out["candidate"].astype(str)
    out["reclaim_covered"] = out.get("reclaim_bar_covered", False)
    out["reclaim_covered"] = out["reclaim_covered"].fillna(False).astype(bool)
    out["entry_pre_rank_covered"] = (
        out.get("shock_bar_covered", False).fillna(False).astype(bool)
        | out.get("pullback_window_covered", False).fillna(False).astype(bool)
        | out.get("reclaim_bar_covered", False).fillna(False).astype(bool)
    )
    out["rank_first_come"] = 0.0
    out["rank_reclaim_taker_buy_ratio"] = _num(out, "reclaim_bar_taker_buy_ratio")
    out["rank_reclaim_cvd_delta"] = _num(out, "reclaim_bar_cvd_delta_turnover")
    out["rank_shock_taker_buy_ratio"] = _num(out, "shock_bar_taker_buy_ratio")
    out["rank_shock_cvd_delta"] = _num(out, "shock_bar_cvd_delta_turnover")
    out["rank_large_buy_volume"] = _num(out, "reclaim_bar_large_buy_turnover").fillna(0.0) + _num(
        out, "shock_bar_large_buy_turnover"
    ).fillna(0.0)
    out["rank_sell_pressure_low"] = -(
        _num(out, "reclaim_bar_large_sell_turnover").fillna(0.0)
        + _num(out, "shock_bar_large_sell_turnover").fillna(0.0)
    )
    out["rank_old_cluster"] = _num(out, "cluster_impulse_density", 0.0).fillna(0.0)
    components = [
        out["rank_reclaim_taker_buy_ratio"].rank(pct=True),
        out["rank_reclaim_cvd_delta"].rank(pct=True),
        out["rank_shock_taker_buy_ratio"].rank(pct=True),
        (-_num(out, "reclaim_bar_large_sell_turnover").fillna(0.0)).rank(pct=True),
    ]
    out["rank_orderflow_composite"] = pd.concat(components, axis=1).mean(axis=1).fillna(0.0)
    return out.dropna(subset=["entry_time", "exit_time", "net_return"]).reset_index(drop=True)


def _pool(data: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    candidates = POOLS[pool_name]
    sample = data[data["candidate"].isin(candidates)].copy()
    if pool_name == "P2_CIC1_CIC2_COMBINED" and not sample.empty:
        sample["candidate_priority"] = sample["candidate"].map({"CIC1_FILTERED_MIR1": 2, "CIC2_FILTERED_MIR1": 1}).fillna(0)
        dedupe_cols = [col for col in ["signal_id", "symbol", "entry_time"] if col in sample.columns]
        if dedupe_cols:
            sample = sample.sort_values(dedupe_cols + ["candidate_priority"], ascending=[True] * len(dedupe_cols) + [False])
            sample = sample.drop_duplicates(dedupe_cols, keep="first")
    sample["pool"] = pool_name
    return sample


def _month_cap_expectancy(sample: pd.DataFrame, cap: float = 0.35) -> float:
    if sample.empty:
        return np.nan
    total = _num(sample, "net_return").sum()
    cap_value = total * cap if total > 0 else 0.0
    capped = []
    for _, group in sample.groupby("month", sort=False, dropna=False):
        value = _num(group, "net_return").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped))


def _max_contribution(sample: pd.DataFrame, group_col: str) -> float:
    if sample.empty or group_col not in sample.columns:
        return np.nan
    grouped = sample.groupby(group_col, sort=False, dropna=False)["net_return"].sum()
    total = grouped.sum()
    return float((grouped / total).abs().max()) if total else np.nan


def _metrics(selected: pd.DataFrame, skipped: pd.DataFrame, *, pool: str, rule: str, max_positions: int) -> dict[str, object]:
    selected_net = _num(selected, "net_return")
    skipped_net = _num(skipped, "net_return")
    return {
        "ranking_rule": rule,
        "pool": pool,
        "max_positions": max_positions,
        "selected_trades": int(len(selected)),
        "skipped_trades": int(len(skipped)),
        "selected_net20": float(selected_net.mean()) if len(selected_net) else np.nan,
        "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "selected_minus_skipped": float(selected_net.mean() - skipped_net.mean()) if len(selected_net) and len(skipped_net) else np.nan,
        "portfolio_net20": float(selected_net.sum()) if len(selected_net) else 0.0,
        "month_cap35_net20": _month_cap_expectancy(selected),
        "max_symbol_contribution": _max_contribution(selected, "symbol"),
        "reclaim_coverage_rate": float(selected["reclaim_covered"].mean()) if "reclaim_covered" in selected.columns and len(selected) else np.nan,
    }


def _coverage(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        for candidate, group in sample.groupby("candidate", dropna=False):
            row = {
                "pool": pool_name,
                "candidate": candidate,
                "trades": int(len(group)),
                "portfolio_accepted": int(group.get("portfolio_accepted", pd.Series(False, index=group.index)).fillna(False).astype(bool).sum()),
            }
            for window in ["shock_bar", "pullback_window", "reclaim_bar", "entry_bar", "post_entry_15m", "post_entry_1h"]:
                covered = group.get(f"{window}_covered", pd.Series(False, index=group.index)).fillna(False).astype(bool)
                row[f"{window}_coverage_rate"] = float(covered.mean()) if len(covered) else np.nan
                row[f"{window}_covered_trades"] = int(covered.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _bucket_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        if sample.empty:
            continue
        for feature in ORDERFLOW_FEATURES:
            values = _num(sample, feature)
            valid = sample[values.notna()].copy()
            if len(valid) < 10:
                continue
            try:
                valid["bucket"] = pd.qcut(_num(valid, feature), q=min(5, len(valid)), duplicates="drop")
            except ValueError:
                continue
            for bucket, group in valid.groupby("bucket", observed=True):
                rows.append(
                    {
                        "pool": pool_name,
                        "feature": feature,
                        "bucket": str(bucket),
                        "trades": int(len(group)),
                        "net20_avg": float(_num(group, "net_return").mean()),
                        "net20_sum": float(_num(group, "net_return").sum()),
                        "positive_rate": float((_num(group, "net_return") > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _ranking_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_frames = []
    skipped_frames = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        if sample.empty:
            continue
        for rule, score_col in RANKING_RULES.items():
            if score_col not in sample.columns:
                continue
            for max_positions in MAX_POSITIONS:
                selected, skipped = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                row = _metrics(selected, skipped, pool=pool_name, rule=rule, max_positions=max_positions)
                rows.append(row)
                if pool_name == "P2_CIC1_CIC2_COMBINED" and max_positions in [5, 8]:
                    if not selected.empty:
                        selected_frames.append(selected.assign(ranking_rule=rule, pool=pool_name, max_positions=max_positions))
                    if not skipped.empty:
                        skipped_frames.append(skipped.assign(ranking_rule=rule, pool=pool_name, max_positions=max_positions))
    return (
        pd.DataFrame(rows),
        pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _random_comparison(data: pd.DataFrame, permutations: int = 100, seed: int = 907) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        if sample.empty:
            continue
        base = {}
        for max_positions in MAX_POSITIONS:
            random_totals = []
            for _ in range(permutations):
                local = sample.copy()
                local["rank_random"] = rng.normal(size=len(local))
                selected, _ = select_portfolio(local, score_col="rank_random", max_positions=max_positions)
                random_totals.append(float(_num(selected, "net_return").sum()) if not selected.empty else 0.0)
            base[(pool_name, max_positions)] = np.array(random_totals, dtype=float)
        for rule, score_col in RANKING_RULES.items():
            if score_col not in sample.columns:
                continue
            for max_positions in MAX_POSITIONS:
                selected, _ = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                total = float(_num(selected, "net_return").sum()) if not selected.empty else 0.0
                dist = base[(pool_name, max_positions)]
                rows.append(
                    {
                        "pool": pool_name,
                        "ranking_rule": rule,
                        "max_positions": max_positions,
                        "portfolio_net20": total,
                        "random_mean": float(np.mean(dist)),
                        "random_median": float(np.median(dist)),
                        "random_p75": float(np.quantile(dist, 0.75)),
                        "random_p90": float(np.quantile(dist, 0.90)),
                        "random_percentile": float((dist <= total).mean()),
                        "permutations": permutations,
                    }
                )
    return pd.DataFrame(rows)


def _conflict_set_analysis(data: pd.DataFrame) -> pd.DataFrame:
    sample = _pool(data, "P2_CIC1_CIC2_COMBINED")
    if sample.empty:
        return pd.DataFrame()
    rows = []
    sample = sample.copy()
    sample["decision_bucket"] = sample["entry_time"].dt.floor("15min")
    for (bucket, rule), group in sample.groupby(["decision_bucket", "candidate"], dropna=False):
        if len(group) < 2:
            continue
        rows.append(
            {
                "decision_time": bucket,
                "candidate": rule,
                "num_candidates": int(len(group)),
                "oracle_top5_net20": float(_num(group.nlargest(min(5, len(group)), "net_return"), "net_return").sum()),
                "all_candidates_net20": float(_num(group, "net_return").sum()),
                "avg_net20": float(_num(group, "net_return").mean()),
            }
        )
    return pd.DataFrame(rows)


def _post_entry_followthrough(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        if sample.empty:
            continue
        for feature in ["post_entry_15m_cvd_delta_turnover", "post_entry_1h_cvd_delta_turnover"]:
            covered = sample[_num(sample, feature).notna()].copy()
            if covered.empty:
                rows.append({"pool": pool_name, "feature": feature, "covered_trades": 0})
                continue
            covered["positive"] = _num(covered, feature) > 0
            for label, group in covered.groupby("positive", dropna=False):
                rows.append(
                    {
                        "pool": pool_name,
                        "feature": feature,
                        "positive": bool(label),
                        "covered_trades": int(len(group)),
                        "net20_avg": float(_num(group, "net_return").mean()),
                        "positive_trade_rate": float((_num(group, "net_return") > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _write_notes(path: Path, data: pd.DataFrame, coverage: pd.DataFrame, ranking: pd.DataFrame) -> None:
    cic_trades = int(data[data["candidate"].isin(POOLS["P2_CIC1_CIC2_COMBINED"])].shape[0]) if not data.empty else 0
    reclaim_covered = int(data.get("reclaim_covered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not data.empty else 0
    if reclaim_covered >= MIN_DECISION_TRADES:
        status = "candidate_check"
    elif reclaim_covered >= MIN_DIAGNOSTIC_TRADES:
        status = "diagnostic_only"
    else:
        status = "insufficient_orderflow_coverage"
    lines = [
        "# v0.9C Orderflow-Aware Capacity Ranking",
        "",
        f"sample_status: {status}",
        f"cic_pool_trades: {cic_trades}",
        f"reclaim_orderflow_covered_trades: {reclaim_covered}",
        "",
        "This report is shadow-only. It does not change CIC/P2 primary execution and does not allow real-live.",
        "",
        "Promotion requires selected_net20 > skipped_net20, positive random lift, sufficient orderflow coverage, and healthy concentration.",
        "",
    ]
    if coverage.empty or reclaim_covered < MIN_DIAGNOSTIC_TRADES:
        lines.append("Current orderflow coverage is too small for ranking conclusions. Treat all ranking rows as framework smoke tests only.")
    elif not ranking.empty:
        best = ranking.sort_values("portfolio_net20", ascending=False).head(5)
        lines.append("## Best Shadow Rows")
        for row in best.itertuples(index=False):
            lines.append(
                f"- {row.pool} {row.ranking_rule} max={row.max_positions}: "
                f"portfolio_net20={row.portfolio_net20:.4%}, selected_minus_skipped={row.selected_minus_skipped:.4%}."
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v09c_orderflow_capacity_ranking(
    source_root: Path = SOURCE_ROOT,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    data = _prepare(_read_source(source_root))
    coverage = _coverage(data)
    bucket = _bucket_summary(data)
    ranking, selected, skipped = _ranking_summary(data)
    random = _random_comparison(data) if len(data) >= 5 else pd.DataFrame()
    if not ranking.empty and not random.empty:
        ranking = ranking.merge(
            random[["pool", "ranking_rule", "max_positions", "random_percentile"]],
            on=["pool", "ranking_rule", "max_positions"],
            how="left",
        )
    conflict = _conflict_set_analysis(data)
    post = _post_entry_followthrough(data)
    outputs = {
        "orderflow_feature_coverage": report_root / "orderflow_feature_coverage.csv",
        "orderflow_bucket_summary": report_root / "orderflow_bucket_summary.csv",
        "orderflow_ranking_summary": report_root / "orderflow_ranking_summary.csv",
        "selected_vs_skipped_orderflow": report_root / "selected_vs_skipped_orderflow.csv",
        "random_permutation_comparison": report_root / "random_permutation_comparison.csv",
        "conflict_set_orderflow_analysis": report_root / "conflict_set_orderflow_analysis.csv",
        "post_entry_followthrough_diagnostic": report_root / "post_entry_followthrough_diagnostic.csv",
        "selected_trades": report_root / "selected_trades.csv",
        "skipped_trades": report_root / "skipped_trades.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["orderflow_feature_coverage"], index=False)
    bucket.to_csv(outputs["orderflow_bucket_summary"], index=False)
    ranking.to_csv(outputs["orderflow_ranking_summary"], index=False)
    ranking.to_csv(outputs["selected_vs_skipped_orderflow"], index=False)
    random.to_csv(outputs["random_permutation_comparison"], index=False)
    conflict.to_csv(outputs["conflict_set_orderflow_analysis"], index=False)
    post.to_csv(outputs["post_entry_followthrough_diagnostic"], index=False)
    selected.to_csv(outputs["selected_trades"], index=False)
    skipped.to_csv(outputs["skipped_trades"], index=False)
    _write_notes(outputs["candidate_notes"], data, coverage, ranking)
    return outputs


__all__ = ["REPORT_ROOT", "write_v09c_orderflow_capacity_ranking"]
