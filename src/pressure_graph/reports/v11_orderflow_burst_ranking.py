"""v1.1 orderflow burst ranking: can taker-flow rank CIC candidates inside a burst?

Question (collaboration doc sections 4 + 5.2): under capacity limits, selected
CIC trades trail skipped counterfactuals; can pre-entry orderflow (taker buy
ratio, CVD imbalance, large-trade pressure) recover the ranking edge that
price/volume context features could not?

Inputs:
- v0.9D capacity trade cache (year-long CIC1/CIC2 candidate stream).
- Historical event orderflow built by ``pressure_graph.orderflow_history``
  (Binance UM aggTrades archives; the v0.8.1 demand queue lists binance_proxy
  as an accepted source preference).

Leak discipline: ranking scores may only consume PRE-entry windows (shock bar,
pullback window, reclaim bar, pre-entry union). Entry-bar and post-entry
windows are diagnostics only. ``RANK_FEATURES`` is asserted against
``orderflow_history.PRE_ENTRY_WINDOWS`` at import time.

This report changes no paper-live or real-live permissions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.orderflow_history import HISTORY_ROOT, PRE_ENTRY_WINDOWS
from pressure_graph.reports.v07d1 import _signal_id
from pressure_graph.reports.v09b import FOCAL_COST, _pool_trades, select_portfolio
from pressure_graph.reports.v09d import _add_burst_id
from pressure_graph.reports.v10a_cic_basket_portfolio import _portfolio_metrics

REPORT_ROOT = Path("reports/v1_1_orderflow_burst_ranking")
EVENT_ORDERFLOW_PATH = HISTORY_ROOT / "cic_event_orderflow.parquet"

POOL_NAME = "P2_CIC1_CIC2_COMBINED"
STRESS_COSTS = (20.0, 30.0, 50.0)
MAX_POSITIONS_GRID = (5, 8, 10)
BURST_WINDOW = "2h"
MIN_BURST_TRADES = 2
WALK_FORWARD_MIN_TRAIN_MONTHS = 3
LOGISTIC_STEPS = 600
LOGISTIC_LR = 0.1
LOGISTIC_L2 = 1.0

# Scale-free pre-entry features: each entry is (output column, window, stat) or
# a derived combination handled in _attach_rank_features.
RANK_FEATURES = {
    "of_shock_taker_buy_ratio": ("shock_bar", "taker_buy_ratio"),
    "of_shock_imbalance": ("shock_bar", "buy_sell_imbalance"),
    "of_pullback_imbalance": ("pullback_window", "buy_sell_imbalance"),
    "of_reclaim_taker_buy_ratio": ("reclaim_bar", "taker_buy_ratio"),
    "of_reclaim_imbalance": ("reclaim_bar", "buy_sell_imbalance"),
    "of_pre_entry_imbalance": ("pre_entry_all", "buy_sell_imbalance"),
}
DERIVED_FEATURES = ("of_large_buy_share", "of_large_sell_share_neg")
COMPOSITE_COMPONENTS = (
    "of_reclaim_taker_buy_ratio",
    "of_reclaim_imbalance",
    "of_pullback_imbalance",
    "of_large_sell_share_neg",
)
ALL_RANK_COLUMNS = tuple(RANK_FEATURES) + DERIVED_FEATURES + ("of_composite",)

for _column, (_window, _stat) in RANK_FEATURES.items():
    if _window not in PRE_ENTRY_WINDOWS:
        raise AssertionError(f"rank feature {_column} uses non-pre-entry window {_window}")


@dataclass(frozen=True)
class V11Config:
    report_root: Path = REPORT_ROOT
    event_orderflow_path: Path = EVENT_ORDERFLOW_PATH
    burst_window: str = BURST_WINDOW
    min_coverage_ratio: float = 0.5


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _pool_at_cost(trades: pd.DataFrame, cost: float) -> pd.DataFrame:
    pool = _pool_trades(trades, POOL_NAME)
    if pool.empty:
        return pool.copy()
    pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(float(cost))].copy()
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce")
    pool["net_return"] = pd.to_numeric(pool["net_return"], errors="coerce")
    return (
        pool.dropna(subset=["entry_time", "exit_time", "net_return"])
        .sort_values(["entry_time", "symbol", "candidate"])
        .reset_index(drop=True)
    )


def _join_orderflow(pool: pd.DataFrame, orderflow: pd.DataFrame, cfg: V11Config) -> pd.DataFrame:
    out = pool.copy()
    out["signal_id"] = _signal_id(out)
    feature_cols = [col for col in orderflow.columns if col not in {"exchange", "symbol", "candidate", "signal_time", "entry_time"}]
    merged = out.merge(orderflow[feature_cols], on="signal_id", how="left", suffixes=("", "_of"))
    pre_entry_cov = pd.concat(
        [
            pd.to_numeric(merged.get(f"{window}_coverage_ratio"), errors="coerce").fillna(0.0)
            for window in ("shock_bar", "reclaim_bar")
        ],
        axis=1,
    ).min(axis=1)
    merged["of_pre_entry_coverage"] = pre_entry_cov
    merged["of_covered"] = pre_entry_cov >= cfg.min_coverage_ratio
    return merged


def _attach_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, (window, stat) in RANK_FEATURES.items():
        out[column] = _num(out, f"{window}_{stat}")
    pre_turnover = _num(out, "shock_bar_turnover").fillna(0.0) + _num(out, "reclaim_bar_turnover").fillna(0.0)
    large_buy = _num(out, "shock_bar_large_buy_turnover").fillna(0.0) + _num(out, "reclaim_bar_large_buy_turnover").fillna(0.0)
    large_sell = _num(out, "shock_bar_large_sell_turnover").fillna(0.0) + _num(out, "reclaim_bar_large_sell_turnover").fillna(0.0)
    denom = pre_turnover.replace(0.0, np.nan)
    out["of_large_buy_share"] = large_buy / denom
    out["of_large_sell_share_neg"] = -(large_sell / denom)
    uncovered = ~out["of_covered"].fillna(False).astype(bool)
    for column in tuple(RANK_FEATURES) + DERIVED_FEATURES:
        out.loc[uncovered, column] = np.nan
        median = out[column].median()
        out[f"{column}_filled"] = out[column].fillna(median if np.isfinite(median) else 0.0)
    components = [out[f"{column}_filled"].rank(pct=True) for column in COMPOSITE_COMPONENTS]
    out["of_composite"] = pd.concat(components, axis=1).mean(axis=1)
    out["of_composite_filled"] = out["of_composite"].fillna(out["of_composite"].median())
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


def _feature_ic(sample: pd.DataFrame, cfg: V11Config) -> pd.DataFrame:
    covered = sample[sample["of_covered"].fillna(False).astype(bool)].copy()
    bursts = _add_burst_id(covered, cfg.burst_window)
    rows = []
    for column in ALL_RANK_COLUMNS:
        feature = _num(covered, column)
        net = _num(covered, "net_return")
        within_ics = []
        for _, group in bursts.groupby("burst_id", sort=False):
            if len(group) < MIN_BURST_TRADES:
                continue
            ic = _spearman(_num(group, column), _num(group, "net_return"))
            if np.isfinite(ic):
                within_ics.append(ic)
        rows.append(
            {
                "feature": column,
                "covered_trades": int(feature.notna().sum()),
                "global_spearman_ic": _spearman(feature, net),
                "within_burst_mean_ic": float(np.mean(within_ics)) if within_ics else np.nan,
                "within_burst_ic_count": int(len(within_ics)),
                "within_burst_ic_positive_rate": float(np.mean([ic > 0 for ic in within_ics])) if within_ics else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_winrate(sample: pd.DataFrame, cfg: V11Config) -> pd.DataFrame:
    covered = sample[sample["of_covered"].fillna(False).astype(bool)].copy()
    bursts = _add_burst_id(covered, cfg.burst_window)
    rows = []
    for column in ALL_RANK_COLUMNS:
        wins = 0
        ties = 0
        total = 0
        for _, group in bursts.groupby("burst_id", sort=False):
            if len(group) < MIN_BURST_TRADES:
                continue
            feature = _num(group, column).to_numpy()
            net = _num(group, "net_return").to_numpy()
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j or not np.isfinite(feature[i]) or not np.isfinite(feature[j]):
                        continue
                    if feature[i] <= feature[j]:
                        continue
                    total += 1
                    if net[i] > net[j]:
                        wins += 1
                    elif net[i] == net[j]:
                        ties += 1
        rows.append(
            {
                "feature": column,
                "pairs": int(total),
                "higher_feature_win_rate": float(wins / total) if total else np.nan,
                "tie_rate": float(ties / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _feature_quintiles(sample: pd.DataFrame) -> pd.DataFrame:
    covered = sample[sample["of_covered"].fillna(False).astype(bool)].copy()
    rows = []
    for column in ALL_RANK_COLUMNS:
        feature = _num(covered, column)
        net = _num(covered, "net_return")
        mask = feature.notna() & net.notna()
        if int(mask.sum()) < 25:
            continue
        try:
            buckets = pd.qcut(feature[mask], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        for bucket, group_net in net[mask].groupby(buckets):
            rows.append(
                {
                    "feature": column,
                    "quintile": int(bucket) + 1,
                    "trades": int(len(group_net)),
                    "net20_avg": float(group_net.mean()),
                    "net20_median": float(group_net.median()),
                    "hit_rate": float((group_net > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _replay_metrics(
    pool: pd.DataFrame,
    score_col: str,
    rule: str,
    max_positions: int,
    cost: float,
    notes: str = "",
) -> dict[str, object]:
    selected, skipped = select_portfolio(pool, score_col=score_col, max_positions=max_positions)
    metrics = _portfolio_metrics(
        selected,
        skipped,
        architecture="orderflow_capacity_replay",
        pool=POOL_NAME,
        rule=rule,
        max_positions=max_positions,
        notes=notes,
    )
    metrics["cost_single_side_bps"] = float(cost)
    return metrics


def _counterfactual_replays(trades: pd.DataFrame, orderflow: pd.DataFrame, cfg: V11Config) -> pd.DataFrame:
    rows = []
    for cost in STRESS_COSTS:
        pool = _pool_at_cost(trades, cost)
        if pool.empty:
            continue
        pool = _attach_rank_features(_join_orderflow(pool, orderflow, cfg))
        coverage = float(pool["of_covered"].fillna(False).astype(bool).mean())
        pool["rank_first_come_first_served"] = 0.0
        rules = {"R0_first_come": "rank_first_come_first_served"}
        for column in ALL_RANK_COLUMNS:
            rules[f"R_{column}"] = f"{column}_filled"
        for max_positions in MAX_POSITIONS_GRID:
            for rule, score_col in rules.items():
                rows.append(
                    _replay_metrics(
                        pool,
                        score_col,
                        rule,
                        max_positions,
                        cost,
                        notes=f"of_coverage={coverage:.2%}",
                    )
                )
    return pd.DataFrame(rows)


def _standardize(train: np.ndarray, apply: np.ndarray) -> np.ndarray:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (apply - mean) / std


def _fit_pairwise_logistic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(features.shape[1])
    n = len(labels)
    if n == 0:
        return weights
    for _ in range(LOGISTIC_STEPS):
        logits = features @ weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        gradient = features.T @ (probs - labels) / n + LOGISTIC_L2 * weights / n
        weights -= LOGISTIC_LR * gradient
    return weights


def _walk_forward_learned(trades: pd.DataFrame, orderflow: pd.DataFrame, cfg: V11Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _pool_at_cost(trades, FOCAL_COST)
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    pool = _attach_rank_features(_join_orderflow(pool, orderflow, cfg))
    pool["month"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    bursts = _add_burst_id(pool, cfg.burst_window)
    feature_cols = [f"{column}_filled" for column in tuple(RANK_FEATURES) + DERIVED_FEATURES]
    months = sorted(bursts["month"].dropna().unique())
    scores = pd.Series(np.nan, index=bursts.index)
    weight_rows = []
    for idx in range(WALK_FORWARD_MIN_TRAIN_MONTHS, len(months)):
        test_month = months[idx]
        train_months = set(months[:idx])
        train = bursts[bursts["month"].isin(train_months)]
        test_mask = bursts["month"].eq(test_month)
        pairs_x: list[np.ndarray] = []
        pairs_y: list[float] = []
        for _, group in train.groupby("burst_id", sort=False):
            if len(group) < MIN_BURST_TRADES:
                continue
            matrix = group[feature_cols].to_numpy(dtype=float)
            net = _num(group, "net_return").to_numpy(dtype=float)
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j or not np.isfinite(net[i]) or not np.isfinite(net[j]) or net[i] == net[j]:
                        continue
                    pairs_x.append(matrix[i] - matrix[j])
                    pairs_y.append(1.0 if net[i] > net[j] else 0.0)
        if not pairs_x:
            continue
        train_x = np.vstack(pairs_x)
        train_y = np.asarray(pairs_y)
        train_matrix = train[feature_cols].to_numpy(dtype=float)
        norm_x = _standardize(train_matrix, train_x)
        weights = _fit_pairwise_logistic(np.nan_to_num(norm_x), train_y)
        test_matrix = bursts.loc[test_mask, feature_cols].to_numpy(dtype=float)
        scores.loc[test_mask] = np.nan_to_num(_standardize(train_matrix, test_matrix)) @ weights
        weight_rows.append(
            {
                "test_month": test_month,
                "train_pairs": int(len(train_y)),
                **{col: float(w) for col, w in zip(feature_cols, weights, strict=True)},
            }
        )
    bursts["of_learned_wf"] = scores
    scored = bursts[bursts["of_learned_wf"].notna()].copy()
    rows = []
    if not scored.empty:
        scored["rank_first_come_first_served"] = 0.0
        for max_positions in MAX_POSITIONS_GRID:
            rows.append(
                _replay_metrics(scored, "rank_first_come_first_served", "R0_first_come_scored_period", max_positions, FOCAL_COST)
            )
            rows.append(
                _replay_metrics(scored, "of_composite_filled", "R_of_composite_scored_period", max_positions, FOCAL_COST)
            )
            rows.append(
                _replay_metrics(scored, "of_learned_wf", "R_of_learned_walk_forward", max_positions, FOCAL_COST)
            )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def _coverage_join_summary(trades: pd.DataFrame, orderflow: pd.DataFrame, cfg: V11Config) -> pd.DataFrame:
    pool = _pool_at_cost(trades, FOCAL_COST)
    if pool.empty:
        return pd.DataFrame()
    joined = _join_orderflow(pool, orderflow, cfg)
    rows = []
    for candidate, sample in joined.groupby("candidate", dropna=False):
        status = sample["mapping_status"].fillna("").astype(str)
        rows.append(
            {
                "candidate": candidate,
                "pool_trades": int(len(sample)),
                "orderflow_rows_matched": int(status.ne("").sum()),
                "mapped": int(status.eq("mapped").sum()),
                "unlisted": int(status.eq("unlisted").sum()),
                "pre_entry_covered": int(sample["of_covered"].fillna(False).astype(bool).sum()),
                "pre_entry_covered_rate": float(sample["of_covered"].fillna(False).astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    report_root: Path,
    coverage: pd.DataFrame,
    ic: pd.DataFrame,
    winrate: pd.DataFrame,
    replays: pd.DataFrame,
    walk_forward: pd.DataFrame,
) -> None:
    lines = [
        "# v1.1 Orderflow Burst Ranking",
        "",
        "Purpose: test whether pre-entry taker-flow features rank same-burst CIC",
        "candidates better than first-come selection (collab doc sections 4 + 5.2).",
        "Tape source: Binance UM aggTrades archives (binance_proxy preference).",
        "This report changes no paper-live or real-live permissions.",
        "",
    ]
    if not coverage.empty:
        lines.append("## Coverage")
        for row in coverage.itertuples(index=False):
            lines.append(
                f"- {row.candidate}: pool={row.pool_trades}, mapped={row.mapped}, "
                f"pre-entry covered={row.pre_entry_covered} ({row.pre_entry_covered_rate:.1%})."
            )
        lines.append("")
    if not ic.empty:
        lines.append("## Feature IC (within-burst)")
        ordered = ic.sort_values("within_burst_mean_ic", ascending=False)
        for row in ordered.itertuples(index=False):
            lines.append(
                f"- {row.feature}: within-burst IC={row.within_burst_mean_ic:.3f} "
                f"(bursts={row.within_burst_ic_count}, positive={row.within_burst_ic_positive_rate:.1%}), "
                f"global IC={row.global_spearman_ic:.3f}."
            )
        lines.append("")
    if not winrate.empty:
        lines.append("## Same-burst pairwise win rate")
        for row in winrate.sort_values("higher_feature_win_rate", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.feature}: win_rate={row.higher_feature_win_rate:.1%} over {row.pairs} pairs."
            )
        lines.append("")
    if not replays.empty:
        focal = replays[
            replays["cost_single_side_bps"].eq(FOCAL_COST)
            & replays["max_positions"].astype(str).eq("8")
        ].copy()
        if not focal.empty:
            lines.append("## P2 max8 counterfactual replays (20bp)")
            baseline = focal[focal["rule"].eq("R0_first_come")]
            base_net = float(baseline["portfolio_net20"].iloc[0]) if not baseline.empty else np.nan
            for row in focal.sort_values("portfolio_net20", ascending=False).itertuples(index=False):
                delta = row.portfolio_net20 - base_net if np.isfinite(base_net) else np.nan
                lines.append(
                    f"- {row.rule}: portfolio_net20={row.portfolio_net20:.4%} "
                    f"(delta vs R0={delta:+.4%}), selected-skipped gap={row.selected_minus_skipped_net20:.4%}."
                )
            lines.append("")
    if not walk_forward.empty:
        lines.append("## Walk-forward learned ranker (scored period only)")
        for row in walk_forward.sort_values(["max_positions", "rule"]).itertuples(index=False):
            lines.append(
                f"- max={row.max_positions} {row.rule}: portfolio_net20={row.portfolio_net20:.4%}, "
                f"selected={row.selected_trades}, skipped={row.skipped_trades}."
            )
        lines.append("")
    lines.extend(
        [
            "## Discipline",
            "- Ranking features consume pre-entry windows only; entry/post windows are diagnostics.",
            "- Missing orderflow coverage falls back to feature medians (neutral rank).",
            "- No paper-live primary or real-live permission changes from this report.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v11_orderflow_burst_ranking(
    trades: pd.DataFrame,
    cfg: V11Config = V11Config(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    if not cfg.event_orderflow_path.exists():
        raise FileNotFoundError(
            f"event orderflow not found: {cfg.event_orderflow_path} (run collect-orderflow-history first)"
        )
    orderflow = read_parquet(cfg.event_orderflow_path)
    pool = _pool_at_cost(trades, FOCAL_COST)
    enriched = _attach_rank_features(_join_orderflow(pool, orderflow, cfg))

    coverage = _coverage_join_summary(trades, orderflow, cfg)
    ic = _feature_ic(enriched, cfg)
    winrate = _pairwise_winrate(enriched, cfg)
    quintiles = _feature_quintiles(enriched)
    replays = _counterfactual_replays(trades, orderflow, cfg)
    walk_forward, wf_weights = _walk_forward_learned(trades, orderflow, cfg)

    outputs = {
        "coverage_join_summary": report_root / "coverage_join_summary.csv",
        "feature_ic": report_root / "feature_ic.csv",
        "pairwise_winrate": report_root / "pairwise_winrate.csv",
        "feature_quintiles": report_root / "feature_quintiles.csv",
        "counterfactual_replays": report_root / "counterfactual_replays.csv",
        "walkforward_replay": report_root / "walkforward_replay.csv",
        "walkforward_weights": report_root / "walkforward_weights.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["coverage_join_summary"], index=False)
    ic.to_csv(outputs["feature_ic"], index=False)
    winrate.to_csv(outputs["pairwise_winrate"], index=False)
    quintiles.to_csv(outputs["feature_quintiles"], index=False)
    replays.to_csv(outputs["counterfactual_replays"], index=False)
    walk_forward.to_csv(outputs["walkforward_replay"], index=False)
    wf_weights.to_csv(outputs["walkforward_weights"], index=False)
    _write_notes(report_root, coverage, ic, winrate, replays, walk_forward)
    return outputs


__all__ = [
    "ALL_RANK_COLUMNS",
    "EVENT_ORDERFLOW_PATH",
    "RANK_FEATURES",
    "REPORT_ROOT",
    "V11Config",
    "write_v11_orderflow_burst_ranking",
]
