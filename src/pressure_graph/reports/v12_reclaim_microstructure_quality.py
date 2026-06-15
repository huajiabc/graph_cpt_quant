"""v1.2 reclaim microstructure quality.

v1.1 rejected 15m aggregate orderflow as a burst selector. This report asks a
narrower question: do pre-entry micro features around the reclaim leg itself
separate stronger CIC candidates inside the same burst?

The report is attribution only. It does not change paper-live, shadow selector,
or real-live permissions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v07d1 import _signal_id
from pressure_graph.reports.v09b import FOCAL_COST, _month_cap_expectancy, _pool_trades
from pressure_graph.reports.v09d import _add_burst_id
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig, _load_or_build_trades
from pressure_graph.reports.v11_orderflow_burst_ranking import EVENT_ORDERFLOW_PATH


REPORT_ROOT = Path("reports/v1_2_reclaim_microstructure_quality")
POOL_NAME = "P2_CIC1_CIC2_COMBINED"
BURST_WINDOW = "2h"
MIN_BURST_TRADES = 2
PAIRWISE_PROMOTION_THRESHOLD = 0.55

PRE_ENTRY_WINDOWS = ("shock_bar", "pullback_window", "reclaim_bar", "pre_entry_all")
POST_ENTRY_WINDOWS = ("entry_bar", "post_entry_1h")

PRE_MICRO_FEATURES = (
    "micro_fast_reclaim",
    "micro_reclaim_taker_buy_ratio",
    "micro_reclaim_imbalance",
    "micro_reclaim_cvd_intensity",
    "micro_reclaim_large_buy_share",
    "micro_reclaim_large_sell_share_neg",
    "micro_pullback_sell_pressure_neg",
    "micro_cvd_reversal",
    "micro_reclaim_vs_shock",
    "micro_pre_entry_imbalance",
    "micro_reclaim_trade_intensity",
)

POST_DIAGNOSTIC_FEATURES = (
    "diag_entry_bar_imbalance",
    "diag_entry_bar_taker_buy_ratio",
    "diag_post_entry_1h_imbalance",
    "diag_post_entry_1h_taker_buy_ratio",
    "diag_post_entry_1h_cvd_intensity",
)


@dataclass(frozen=True)
class V12Config:
    report_root: Path = REPORT_ROOT
    event_orderflow_path: Path = EVENT_ORDERFLOW_PATH
    burst_window: str = BURST_WINDOW
    min_coverage_ratio: float = 0.5
    v10a: V10AConfig = V10AConfig()


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0.0, np.nan)
    return numerator / denom


def _pool_at_cost(trades: pd.DataFrame, cost: float = FOCAL_COST) -> pd.DataFrame:
    pool = _pool_trades(trades, POOL_NAME)
    if pool.empty:
        return pool.copy()
    pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(float(cost))].copy()
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True, errors="coerce")
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce")
    pool["net_return"] = pd.to_numeric(pool["net_return"], errors="coerce")
    return (
        pool.dropna(subset=["signal_time", "entry_time", "exit_time", "net_return"])
        .sort_values(["entry_time", "symbol", "candidate"])
        .reset_index(drop=True)
    )


def _join_orderflow(pool: pd.DataFrame, orderflow: pd.DataFrame, cfg: V12Config) -> pd.DataFrame:
    out = pool.copy()
    if "signal_id" not in out.columns:
        out["signal_id"] = _signal_id(out)
    feature_cols = [
        col
        for col in orderflow.columns
        if col not in {"exchange", "symbol", "candidate", "signal_time", "entry_time"}
    ]
    merged = out.merge(orderflow[feature_cols], on="signal_id", how="left", suffixes=("", "_of"))
    pre_cov = pd.concat(
        [
            pd.to_numeric(merged.get(f"{window}_coverage_ratio"), errors="coerce").fillna(0.0)
            for window in ("shock_bar", "reclaim_bar")
        ],
        axis=1,
    ).min(axis=1)
    merged["micro_pre_entry_coverage"] = pre_cov
    merged["micro_pre_entry_covered"] = pre_cov >= cfg.min_coverage_ratio
    return merged


def _attach_micro_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    signal = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    entry = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["time_to_reclaim_minutes"] = (entry - signal).dt.total_seconds() / 60.0
    out["micro_fast_reclaim"] = -pd.to_numeric(out["time_to_reclaim_minutes"], errors="coerce")

    reclaim_turnover = _num(out, "reclaim_bar_turnover")
    pullback_turnover = _num(out, "pullback_window_turnover")
    shock_imbalance = _num(out, "shock_bar_buy_sell_imbalance")
    pullback_imbalance = _num(out, "pullback_window_buy_sell_imbalance")
    reclaim_imbalance = _num(out, "reclaim_bar_buy_sell_imbalance")

    out["micro_reclaim_taker_buy_ratio"] = _num(out, "reclaim_bar_taker_buy_ratio")
    out["micro_reclaim_imbalance"] = reclaim_imbalance
    out["micro_reclaim_cvd_intensity"] = _safe_divide(_num(out, "reclaim_bar_cvd_delta_turnover"), reclaim_turnover)
    out["micro_reclaim_large_buy_share"] = _safe_divide(_num(out, "reclaim_bar_large_buy_turnover"), reclaim_turnover)
    out["micro_reclaim_large_sell_share_neg"] = -_safe_divide(_num(out, "reclaim_bar_large_sell_turnover"), reclaim_turnover)
    out["micro_pullback_sell_pressure_neg"] = -_safe_divide(_num(out, "pullback_window_large_sell_turnover"), pullback_turnover)
    out["micro_cvd_reversal"] = reclaim_imbalance - pullback_imbalance
    out["micro_reclaim_vs_shock"] = reclaim_imbalance - shock_imbalance
    out["micro_pre_entry_imbalance"] = _num(out, "pre_entry_all_buy_sell_imbalance")
    out["micro_reclaim_trade_intensity"] = np.log1p(_num(out, "reclaim_bar_trade_count"))

    out["diag_entry_bar_imbalance"] = _num(out, "entry_bar_buy_sell_imbalance")
    out["diag_entry_bar_taker_buy_ratio"] = _num(out, "entry_bar_taker_buy_ratio")
    out["diag_post_entry_1h_imbalance"] = _num(out, "post_entry_1h_buy_sell_imbalance")
    out["diag_post_entry_1h_taker_buy_ratio"] = _num(out, "post_entry_1h_taker_buy_ratio")
    out["diag_post_entry_1h_cvd_intensity"] = _safe_divide(
        _num(out, "post_entry_1h_cvd_delta_turnover"),
        _num(out, "post_entry_1h_turnover"),
    )

    uncovered = ~out["micro_pre_entry_covered"].fillna(False).astype(bool)
    for feature in PRE_MICRO_FEATURES:
        out.loc[uncovered, feature] = np.nan
    return out


def _spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return np.nan
    xr = x[mask].rank()
    yr = y[mask].rank()
    xv = xr - xr.mean()
    yv = yr - yr.mean()
    denom = float(np.sqrt((xv**2).sum() * (yv**2).sum()))
    return float((xv * yv).sum() / denom) if denom else np.nan


def _coverage_summary(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in sample.groupby("candidate", dropna=False):
        status = group.get("mapping_status", pd.Series("", index=group.index)).fillna("").astype(str)
        reclaim_covered = group.get("reclaim_bar_covered", pd.Series(False, index=group.index))
        entry_covered = group.get("entry_bar_covered", pd.Series(False, index=group.index))
        post_covered = group.get("post_entry_1h_covered", pd.Series(False, index=group.index))
        rows.append(
            {
                "candidate": candidate,
                "trades": int(len(group)),
                "mapped": int(status.eq("mapped").sum()),
                "unlisted": int(status.eq("unlisted").sum()),
                "pre_entry_covered": int(group["micro_pre_entry_covered"].fillna(False).astype(bool).sum()),
                "pre_entry_covered_rate": float(group["micro_pre_entry_covered"].fillna(False).astype(bool).mean()),
                "reclaim_covered_rate": float(reclaim_covered.fillna(False).astype(bool).mean()),
                "entry_bar_covered_rate": float(entry_covered.fillna(False).astype(bool).mean()),
                "post_entry_1h_covered_rate": float(post_covered.fillna(False).astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def _feature_summary(sample: pd.DataFrame, cfg: V12Config) -> pd.DataFrame:
    covered = sample[sample["micro_pre_entry_covered"].fillna(False).astype(bool)].copy()
    bursts = _add_burst_id(covered, cfg.burst_window)
    rows = []
    for feature in PRE_MICRO_FEATURES:
        values = _num(covered, feature)
        net = _num(covered, "net_return")
        within_ics = []
        for _, group in bursts.groupby("burst_id", sort=False):
            if len(group) < MIN_BURST_TRADES:
                continue
            ic = _spearman(_num(group, feature), _num(group, "net_return"))
            if np.isfinite(ic):
                within_ics.append(ic)
        q5_minus_q1 = np.nan
        mask = values.notna() & net.notna()
        if int(mask.sum()) >= 25:
            try:
                buckets = pd.qcut(values[mask], 5, labels=False, duplicates="drop")
                grouped = net[mask].groupby(buckets)
                if len(grouped) >= 2:
                    means = grouped.mean()
                    q5_minus_q1 = float(means.iloc[-1] - means.iloc[0])
            except ValueError:
                pass
        rows.append(
            {
                "feature": feature,
                "covered_trades": int(values.notna().sum()),
                "global_spearman_ic": _spearman(values, net),
                "within_burst_mean_ic": float(np.mean(within_ics)) if within_ics else np.nan,
                "within_burst_ic_count": int(len(within_ics)),
                "within_burst_ic_positive_rate": float(np.mean([ic > 0 for ic in within_ics])) if within_ics else np.nan,
                "q5_minus_q1_net20": q5_minus_q1,
            }
        )
    return pd.DataFrame(rows)


def _bucket_summary(sample: pd.DataFrame) -> pd.DataFrame:
    covered = sample[sample["micro_pre_entry_covered"].fillna(False).astype(bool)].copy()
    rows = []
    for feature in PRE_MICRO_FEATURES:
        values = _num(covered, feature)
        net = _num(covered, "net_return")
        mask = values.notna() & net.notna()
        if int(mask.sum()) < 25:
            continue
        try:
            buckets = pd.qcut(values[mask], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        local = covered.loc[mask].copy()
        local["_bucket"] = buckets
        for bucket, group in local.groupby("_bucket", sort=True):
            group_net = pd.to_numeric(group["net_return"], errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "bucket": f"q{int(bucket) + 1}",
                    "bucket_direction": "q5_highest_feature",
                    "trades": int(len(group)),
                    "net20_avg": float(group_net.mean()),
                    "net20_median": float(group_net.median()),
                    "hit_rate": float((group_net > 0).mean()),
                    "month_cap35_net20": _month_cap_expectancy(group),
                    "tp_rate": float(group.get("exit_reason", pd.Series(dtype=str)).astype(str).str.startswith("tp").mean()),
                    "sl_rate": float(group.get("exit_reason", pd.Series(dtype=str)).astype(str).str.startswith("sl").mean()),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_winrate(sample: pd.DataFrame, cfg: V12Config) -> pd.DataFrame:
    covered = sample[sample["micro_pre_entry_covered"].fillna(False).astype(bool)].copy()
    bursts = _add_burst_id(covered, cfg.burst_window)
    rows = []
    for feature in PRE_MICRO_FEATURES:
        wins = 0
        losses = 0
        ties = 0
        total = 0
        competitive_bursts = 0
        for _, group in bursts.groupby("burst_id", sort=False):
            if len(group) < MIN_BURST_TRADES:
                continue
            feature_values = _num(group, feature).to_numpy(dtype=float)
            net = _num(group, "net_return").to_numpy(dtype=float)
            local_pairs = 0
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j or not np.isfinite(feature_values[i]) or not np.isfinite(feature_values[j]):
                        continue
                    if feature_values[i] <= feature_values[j]:
                        continue
                    total += 1
                    local_pairs += 1
                    if net[i] > net[j]:
                        wins += 1
                    elif net[i] < net[j]:
                        losses += 1
                    else:
                        ties += 1
            if local_pairs:
                competitive_bursts += 1
        rows.append(
            {
                "feature": feature,
                "competitive_bursts": competitive_bursts,
                "pairs": int(total),
                "higher_feature_win_rate": float(wins / total) if total else np.nan,
                "higher_feature_loss_rate": float(losses / total) if total else np.nan,
                "tie_rate": float(ties / total) if total else np.nan,
                "passes_55pct_threshold": bool(total and wins / total >= PAIRWISE_PROMOTION_THRESHOLD),
            }
        )
    return pd.DataFrame(rows)


def _post_followthrough_diagnostic(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in POST_DIAGNOSTIC_FEATURES:
        values = _num(sample, feature)
        net = _num(sample, "net_return")
        mask = values.notna() & net.notna()
        covered = int(mask.sum())
        row = {
            "feature": feature,
            "covered_trades": covered,
            "global_spearman_ic": _spearman(values, net),
            "winning_trade_avg": float(values[mask & (net > 0)].mean()) if covered else np.nan,
            "losing_trade_avg": float(values[mask & (net <= 0)].mean()) if covered else np.nan,
        }
        rows.append(row)
    for pre_feature in PRE_MICRO_FEATURES:
        pre = _num(sample, pre_feature)
        post = _num(sample, "diag_post_entry_1h_imbalance")
        rows.append(
            {
                "feature": f"{pre_feature}_vs_post_entry_1h_imbalance",
                "covered_trades": int((pre.notna() & post.notna()).sum()),
                "global_spearman_ic": _spearman(pre, post),
                "winning_trade_avg": np.nan,
                "losing_trade_avg": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    report_root: Path,
    coverage: pd.DataFrame,
    feature_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    lines = [
        "# v1.2 Reclaim Microstructure Quality",
        "",
        "Purpose: test whether finer reclaim-path microstructure separates stronger CIC candidates inside the same burst.",
        "Tape source: historical Binance UM aggTrades proxy orderflow. This remains diagnostic and does not change paper-live or real-live permissions.",
        "",
        "## Coverage",
    ]
    if coverage.empty:
        lines.append("- No coverage rows.")
    else:
        for row in coverage.itertuples(index=False):
            lines.append(
                f"- {row.candidate}: trades={row.trades}, mapped={row.mapped}, "
                f"pre-entry covered={row.pre_entry_covered} ({row.pre_entry_covered_rate:.1%})."
            )
    if not feature_summary.empty:
        lines.extend(["", "## Feature IC"])
        for row in feature_summary.sort_values("within_burst_mean_ic", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.feature}: within-burst IC={row.within_burst_mean_ic:.3f}, "
                f"q5-q1={row.q5_minus_q1_net20:.4%}, covered={row.covered_trades}."
            )
    if not pairwise.empty:
        best = pairwise.sort_values("higher_feature_win_rate", ascending=False).head(1).iloc[0]
        lines.extend(
            [
                "",
                "## Pairwise Gate",
                f"- Best feature: {best.feature}, pairwise win rate={best.higher_feature_win_rate:.1%} over {int(best.pairs)} pairs.",
                f"- Promotion threshold: {PAIRWISE_PROMOTION_THRESHOLD:.0%} within-burst pairwise win rate.",
            ]
        )
        if bool(best.passes_55pct_threshold):
            lines.append("- Result: at least one micro feature passes the research threshold; validate before any shadow selector.")
        else:
            lines.append("- Result: no micro feature passes threshold; keep as diagnostic only.")
    lines.extend(
        [
            "",
            "## Discipline",
            "- Selector-grade features use pre-entry windows only.",
            "- Entry/post-entry windows are diagnostics for future slot-management research.",
            "- No primary, shadow selector, or real-live permission change from this report.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v12_reclaim_microstructure_quality(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V12Config = V12Config(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    trades = _load_or_build_trades(feature_path, instruments, config, report_root, cfg.v10a)
    return write_v12_reclaim_microstructure_quality_from_trades(trades, cfg)


def write_v12_reclaim_microstructure_quality_from_trades(
    trades: pd.DataFrame,
    cfg: V12Config = V12Config(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    if not cfg.event_orderflow_path.exists():
        raise FileNotFoundError(
            f"event orderflow not found: {cfg.event_orderflow_path} (run collect-orderflow-history first)"
        )
    pool = _pool_at_cost(trades, FOCAL_COST)
    if pool.empty:
        raise ValueError("No P2 CIC trades available for v1.2 reclaim microstructure quality.")
    orderflow = read_parquet(cfg.event_orderflow_path)
    enriched = _attach_micro_features(_join_orderflow(pool, orderflow, cfg))

    coverage = _coverage_summary(enriched)
    feature_summary = _feature_summary(enriched, cfg)
    buckets = _bucket_summary(enriched)
    pairwise = _pairwise_winrate(enriched, cfg)
    post_diag = _post_followthrough_diagnostic(enriched)

    outputs = {
        "micro_coverage_summary": report_root / "micro_coverage_summary.csv",
        "micro_feature_summary": report_root / "micro_feature_summary.csv",
        "micro_bucket_summary": report_root / "micro_bucket_summary.csv",
        "micro_pairwise_winrate": report_root / "micro_pairwise_winrate.csv",
        "post_reclaim_followthrough_diagnostic": report_root / "post_reclaim_followthrough_diagnostic.csv",
        "micro_enriched_sample": report_root / "micro_enriched_sample.parquet",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["micro_coverage_summary"], index=False)
    feature_summary.to_csv(outputs["micro_feature_summary"], index=False)
    buckets.to_csv(outputs["micro_bucket_summary"], index=False)
    pairwise.to_csv(outputs["micro_pairwise_winrate"], index=False)
    post_diag.to_csv(outputs["post_reclaim_followthrough_diagnostic"], index=False)
    enriched.to_parquet(outputs["micro_enriched_sample"], index=False)
    _write_notes(report_root, coverage, feature_summary, pairwise)
    return outputs


__all__ = [
    "PRE_MICRO_FEATURES",
    "REPORT_ROOT",
    "V12Config",
    "write_v12_reclaim_microstructure_quality",
    "write_v12_reclaim_microstructure_quality_from_trades",
]
