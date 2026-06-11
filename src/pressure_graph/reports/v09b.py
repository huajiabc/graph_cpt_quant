from __future__ import annotations

import gc
import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07b import TOP_N
from pressure_graph.reports.v07c2 import _month_setup
from pressure_graph.reports.v07d import _simulate_candidate_exit
from pressure_graph.reports.v07d1 import FROZEN_CANDIDATES, _signal_id
from pressure_graph.reports.v09a import REPORT_ROOT as V09A_REPORT_ROOT
from pressure_graph.reports.v09a import add_cluster_graph_features
from pressure_graph.reports.v07d1 import E0_VOL_REGIME


REPORT_ROOT = Path("reports/v0_9b_portfolio_ranking")
RANDOM_PERMUTATIONS = 50
FOCAL_COST = 20.0


def _release_process_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


@dataclass(frozen=True)
class PoolSpec:
    pool: str
    description: str


POOLS = [
    PoolSpec("P0_CIC1_ONLY", "CIC1-filtered MIR1 only."),
    PoolSpec("P1_CIC2_ONLY", "Broader CIC2-filtered MIR1 only."),
    PoolSpec("P2_CIC1_CIC2_COMBINED", "CIC1 and CIC2 combined, de-duplicated by signal."),
    PoolSpec("P3_MIR1_RAW_WITH_CIC_LABELS", "MIR1 raw candidate pool with CIC labels as ranking features."),
]

RANKING_RULES = [
    "first_come_first_served",
    "beta_extreme_strength_high",
    "local_volume_shock_strength_high",
    "market_impulse_density_high",
    "cluster_impulse_density_high",
    "liquidity_high",
    "reclaim_quality_high",
    "cic1_label_first",
    "cic2_label_first",
    "composite_simple",
    "reverse_beta_extreme_strength",
]

RANKING_SCORE_COLUMNS = {
    "first_come_first_served": "rank_first_come_first_served",
    "beta_extreme_strength_high": "rank_beta_extreme_strength",
    "local_volume_shock_strength_high": "rank_local_volume_shock_strength",
    "market_impulse_density_high": "rank_market_impulse_density",
    "cluster_impulse_density_high": "rank_cluster_impulse_density",
    "liquidity_high": "rank_liquidity",
    "reclaim_quality_high": "rank_reclaim_quality",
    "cic1_label_first": "rank_cic1_label",
    "cic2_label_first": "rank_cic2_label",
    "composite_simple": "rank_composite_simple",
    "reverse_beta_extreme_strength": "rank_reverse_beta_extreme_strength",
}

MAX_POSITIONS = [1, 2, 3, 5, 8, 10, 10_000]
RANDOM_MAX_POSITIONS = [1, 2, 3, 5, 8, 10]


def _base_signal_id(frame: pd.DataFrame) -> pd.Series:
    signal_time = pd.to_datetime(frame["signal_time"], utc=True, errors="coerce").astype(str)
    return frame["exchange"].astype(str) + "|" + frame["symbol"].astype(str) + "|" + signal_time


def _safe_numeric(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _read_v09a_membership() -> pd.DataFrame:
    path = V09A_REPORT_ROOT / "cluster_membership.csv"
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path)
    if "month_start" in out.columns:
        out["month_start"] = pd.to_datetime(out["month_start"], utc=True, errors="coerce")
    return out


def _membership_for_month(membership: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    if membership.empty or "month_start" not in membership.columns:
        return pd.DataFrame()
    month = pd.Timestamp(month_start)
    return membership[pd.to_datetime(membership["month_start"], utc=True, errors="coerce").eq(month)].copy()


def _add_cluster_context_if_available(
    sim_window: pd.DataFrame,
    membership: pd.DataFrame,
    month_start: pd.Timestamp,
) -> pd.DataFrame:
    local_membership = _membership_for_month(membership, month_start)
    if local_membership.empty:
        out = sim_window.copy()
        out["cluster_impulse_density"] = np.nan
        out["cluster_size"] = np.nan
        out["cluster_id"] = ""
        return out
    return add_cluster_graph_features(sim_window, local_membership)


def _stream_frozen_trades(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v09b_trades_tmp.csv"
    signal_path = report_root / "_v09b_signals_tmp.csv"
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    membership = _read_v09a_membership()
    wrote_trades = False
    wrote_signals = False
    months = sorted(pd.to_datetime(rank30["month_start"], utc=True, errors="coerce").dropna().drop_duplicates().tolist())
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        sim_window, _, symbols = _month_setup(feature_path, rank30, rank90, regime, config, month_start, next_month)
        if sim_window.empty:
            continue
        sim_window = _add_cluster_context_if_available(sim_window, membership, month_start)
        trade_frames: list[pd.DataFrame] = []
        signal_rows: list[dict[str, object]] = []
        for candidate in FROZEN_CANDIDATES:
            trades, signals = _simulate_candidate_exit(sim_window, candidate, E0_VOL_REGIME, config)
            signal_rows.append({"candidate": candidate.candidate, "signals": signals})
            if not trades.empty:
                trades["candidate"] = candidate.candidate
                trade_frames.append(trades)
        month_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        if not month_trades.empty:
            month_trades["signal_id"] = _signal_id(month_trades)
            month_trades["base_signal_id"] = _base_signal_id(month_trades)
            month_trades.to_csv(trade_path, mode="a", header=not wrote_trades, index=False)
            wrote_trades = True
        month_signals = pd.DataFrame(signal_rows)
        if not month_signals.empty:
            month_signals.to_csv(signal_path, mode="a", header=not wrote_signals, index=False)
            wrote_signals = True
        print(f"v0.9B month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del sim_window, trade_frames, month_trades, month_signals
        _release_process_memory()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    if not signals.empty:
        signals = signals.groupby("candidate", as_index=False, sort=False, dropna=False)["signals"].sum()
    return trades, signals


def _prepare_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    if out.empty:
        return out
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out["month"] = out["signal_time"].dt.strftime("%Y-%m")
    out["base_signal_id"] = _base_signal_id(out)
    bucket = out.get("c2_beta_extension_bucket", pd.Series("unknown", index=out.index)).astype(str)
    out["is_cic1"] = bucket.eq("beta_extreme_overextended")
    out["is_cic2"] = bucket.isin(["beta_extended", "beta_extreme_overextended"])
    out["holding_minutes"] = (out["exit_time"] - out["entry_time"]).dt.total_seconds() / 60.0
    out["rank_beta_extreme_strength"] = _safe_numeric(out, "c2_beta_extension_score")
    out["rank_local_volume_shock_strength"] = _safe_numeric(out, "volume_z_1h")
    out["rank_market_impulse_density"] = _safe_numeric(out, "volume_impulse_density")
    out["rank_cluster_impulse_density"] = _safe_numeric(out, "cluster_impulse_density")
    out["rank_liquidity"] = -_safe_numeric(out, "dynamic_all_rank", 999.0)
    out["rank_reclaim_quality"] = -_safe_numeric(out, "bars_from_signal_to_entry", 999.0)
    out["rank_cic1_label"] = out["is_cic1"].astype(float)
    out["rank_cic2_label"] = out["is_cic2"].astype(float)
    out["rank_reverse_beta_extreme_strength"] = -out["rank_beta_extreme_strength"]
    components = [
        out["rank_beta_extreme_strength"].rank(pct=True),
        out["rank_local_volume_shock_strength"].rank(pct=True),
        out["rank_market_impulse_density"].rank(pct=True),
        out["rank_liquidity"].rank(pct=True),
    ]
    if out["rank_cluster_impulse_density"].notna().any():
        components.append(out["rank_cluster_impulse_density"].rank(pct=True))
    out["rank_composite_simple"] = pd.concat(components, axis=1).mean(axis=1)
    out["rank_first_come_first_served"] = 0.0
    return out


def _dedupe_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    out = pool.copy()
    out["candidate_priority"] = np.select(
        [out["candidate"].astype(str).eq("CIC1_beta_extreme"), out["candidate"].astype(str).eq("CIC2_beta_broad")],
        [3.0, 2.0],
        default=1.0,
    )
    out = out.sort_values(["base_signal_id", "cost_single_side_bps", "candidate_priority"], ascending=[True, True, False])
    return out.drop_duplicates(["base_signal_id", "cost_single_side_bps"], keep="first").copy()


def _pool_trades(trades: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    if pool_name == "P0_CIC1_ONLY":
        out = trades[trades["candidate"].astype(str).eq("CIC1_beta_extreme")].copy()
    elif pool_name == "P1_CIC2_ONLY":
        out = trades[trades["candidate"].astype(str).eq("CIC2_beta_broad")].copy()
    elif pool_name == "P2_CIC1_CIC2_COMBINED":
        out = trades[trades["candidate"].astype(str).isin(["CIC1_beta_extreme", "CIC2_beta_broad"])].copy()
        out = _dedupe_pool(out)
    elif pool_name == "P3_MIR1_RAW_WITH_CIC_LABELS":
        out = trades[trades["candidate"].astype(str).eq("MIR1_reference")].copy()
    else:
        raise KeyError(pool_name)
    out["pool"] = pool_name
    return out


def _max_contribution(sample: pd.DataFrame, group_col: str) -> float:
    if sample.empty or group_col not in sample.columns:
        return np.nan
    grouped = sample.groupby(group_col, sort=False, dropna=False)["net_return"].sum()
    total = grouped.sum()
    return float((grouped / total).abs().max()) if total else np.nan


def _month_cap_expectancy(sample: pd.DataFrame, cap: float = 0.35) -> float:
    if sample.empty:
        return np.nan
    total = pd.to_numeric(sample["net_return"], errors="coerce").sum()
    cap_value = total * cap if total > 0 else 0.0
    capped = []
    for _, group in sample.groupby("month", sort=False, dropna=False):
        value = pd.to_numeric(group["net_return"], errors="coerce").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped) / len(sample)) if len(sample) else np.nan


def select_portfolio(
    trades: pd.DataFrame,
    *,
    score_col: str,
    max_positions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    ranked = trades.copy()
    ranked["entry_time"] = pd.to_datetime(ranked["entry_time"], utc=True, errors="coerce")
    ranked["exit_time"] = pd.to_datetime(ranked["exit_time"], utc=True, errors="coerce")
    ranked["score"] = _safe_numeric(ranked, score_col, 0.0)
    ranked = ranked.sort_values(["entry_time", "score", "symbol"], ascending=[True, False, True])
    active_exits: list[pd.Timestamp] = []
    active_symbols: dict[str, pd.Timestamp] = {}
    selected_rows = []
    skipped_rows = []
    for row in ranked.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active_exits = [exit_time for exit_time in active_exits if exit_time > entry]
        active_symbols = {symbol: exit_time for symbol, exit_time in active_symbols.items() if exit_time > entry}
        payload = row._asdict()
        payload["active_positions_at_decision"] = len(active_exits)
        payload["ranking_score"] = float(getattr(row, "score", np.nan))
        symbol = str(getattr(row, "symbol"))
        if symbol in active_symbols:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if len(active_exits) >= max_positions:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "portfolio_full"
            skipped_rows.append(payload)
            continue
        payload["selection_status"] = "selected"
        payload["skip_reason"] = ""
        selected_rows.append(payload)
        exit_time = pd.Timestamp(row.exit_time)
        active_exits.append(exit_time)
        active_symbols[symbol] = exit_time
    return pd.DataFrame(selected_rows), pd.DataFrame(skipped_rows)


def _portfolio_metrics(
    selected: pd.DataFrame,
    skipped: pd.DataFrame,
    *,
    pool: str,
    ranking: str,
    max_positions: int,
    cost: float,
) -> dict[str, object]:
    net = pd.to_numeric(selected.get("net_return", pd.Series(dtype=float)), errors="coerce")
    skipped_net = pd.to_numeric(skipped.get("net_return", pd.Series(dtype=float)), errors="coerce")
    equity = net.cumsum()
    dd = equity - equity.cummax()
    holding_hours = pd.to_numeric(selected.get("holding_minutes", pd.Series(dtype=float)), errors="coerce").sum() / 60.0
    if not selected.empty:
        start = pd.to_datetime(selected["entry_time"], utc=True, errors="coerce").min()
        end = pd.to_datetime(selected["exit_time"], utc=True, errors="coerce").max()
        period_hours = max((end - start).total_seconds() / 3600.0, 1.0)
    else:
        period_hours = np.nan
    denominator = period_hours * max_positions if max_positions < 10_000 and pd.notna(period_hours) else np.nan
    month_sum = selected.groupby("month", sort=False, dropna=False)["net_return"].sum() if not selected.empty else pd.Series(dtype=float)
    return {
        "pool": pool,
        "ranking": ranking,
        "cost_single_side_bps": cost,
        "max_positions": "unlimited" if max_positions >= 10_000 else max_positions,
        "selected_trades": int(len(selected)),
        "skipped_trades": int(len(skipped)),
        "net_expectancy": float(net.mean()) if len(net) else np.nan,
        "total_net": float(net.sum()) if len(net) else 0.0,
        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
        "worst_month": str(month_sum.idxmin()) if len(month_sum) else "",
        "worst_month_net": float(month_sum.min()) if len(month_sum) else np.nan,
        "month_cap35_net": _month_cap_expectancy(selected),
        "max_month_contribution": _max_contribution(selected, "month"),
        "max_symbol_contribution": _max_contribution(selected, "symbol"),
        "avg_holding_minutes": float(pd.to_numeric(selected.get("holding_minutes", pd.Series(dtype=float)), errors="coerce").mean()) if len(selected) else np.nan,
        "return_per_position_hour": float(net.sum() / holding_hours) if holding_hours else np.nan,
        "return_per_capital_day": float(net.sum() / denominator * 24.0) if denominator else np.nan,
        "capital_utilization": float(holding_hours / denominator) if denominator else np.nan,
        "selected_good_trade_rate": float((net > 0).mean()) if len(net) else np.nan,
        "skipped_good_trade_rate": float((skipped_net > 0).mean()) if len(skipped_net) else np.nan,
        "skipped_bad_trade_rate": float((skipped_net <= 0).mean()) if len(skipped_net) else np.nan,
        "skipped_avg_net": float(skipped_net.mean()) if len(skipped_net) else np.nan,
    }


def _run_deterministic_rankings(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    timeline_frames = []
    skipped_frames = []
    for pool_spec in POOLS:
        pool_all = _pool_trades(trades, pool_spec.pool)
        for cost, cost_pool in pool_all.groupby("cost_single_side_bps", sort=False, dropna=False):
            for ranking in RANKING_RULES:
                score_col = RANKING_SCORE_COLUMNS[ranking]
                if score_col not in cost_pool.columns:
                    continue
                for max_positions in MAX_POSITIONS:
                    selected, skipped = select_portfolio(cost_pool, score_col=score_col, max_positions=max_positions)
                    rows.append(
                        _portfolio_metrics(
                            selected,
                            skipped,
                            pool=pool_spec.pool,
                            ranking=ranking,
                            max_positions=max_positions,
                            cost=float(cost),
                        )
                    )
                    if float(cost) == FOCAL_COST and max_positions in {3, 5}:
                        if not selected.empty:
                            local = selected.copy()
                            local["pool"] = pool_spec.pool
                            local["ranking"] = ranking
                            local["max_positions"] = max_positions
                            timeline_frames.append(local)
                        if not skipped.empty:
                            local = skipped.copy()
                            local["pool"] = pool_spec.pool
                            local["ranking"] = ranking
                            local["max_positions"] = max_positions
                            skipped_frames.append(local)
    return (
        pd.DataFrame(rows),
        pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _run_random_rankings(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(906)
    for pool_spec in POOLS:
        pool_all = _pool_trades(trades, pool_spec.pool)
        for cost, cost_pool in pool_all.groupby("cost_single_side_bps", sort=False, dropna=False):
            for max_positions in RANDOM_MAX_POSITIONS:
                for permutation in range(RANDOM_PERMUTATIONS):
                    local = cost_pool.copy()
                    local["rank_random"] = rng.random(len(local))
                    selected, skipped = select_portfolio(local, score_col="rank_random", max_positions=max_positions)
                    row = _portfolio_metrics(
                        selected,
                        skipped,
                        pool=pool_spec.pool,
                        ranking="random",
                        max_positions=max_positions,
                        cost=float(cost),
                    )
                    row["permutation"] = permutation
                    rows.append(row)
    return pd.DataFrame(rows)


def _attach_random_comparison(summary: pd.DataFrame, random_df: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or random_df.empty:
        return summary
    rows = []
    grouped = random_df.groupby(["pool", "cost_single_side_bps", "max_positions"], sort=False, dropna=False)
    lookup = {key: group for key, group in grouped}
    for item in summary.to_dict("records"):
        key = (item["pool"], item["cost_single_side_bps"], item["max_positions"])
        group = lookup.get(key)
        if group is not None and not group.empty:
            vals = pd.to_numeric(group["net_expectancy"], errors="coerce")
            value = float(item["net_expectancy"]) if pd.notna(item["net_expectancy"]) else np.nan
            item["random_median_net"] = float(vals.median())
            item["random_p75_net"] = float(vals.quantile(0.75))
            item["random_p90_net"] = float(vals.quantile(0.90))
            item["percentile_vs_random"] = float((vals <= value).mean()) if pd.notna(value) else np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def _pool_comparison(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_spec in POOLS:
        pool = _pool_trades(trades, pool_spec.pool)
        for cost, group in pool.groupby("cost_single_side_bps", sort=False, dropna=False):
            net = pd.to_numeric(group["net_return"], errors="coerce")
            rows.append(
                {
                    "pool": pool_spec.pool,
                    "description": pool_spec.description,
                    "cost_single_side_bps": cost,
                    "trades": int(len(group)),
                    "net_expectancy": float(net.mean()) if len(net) else np.nan,
                    "month_cap35_net": _month_cap_expectancy(group),
                    "max_month_contribution": _max_contribution(group, "month"),
                    "max_symbol_contribution": _max_contribution(group, "symbol"),
                    "cic1_rate": float(group["is_cic1"].mean()) if len(group) else np.nan,
                    "cic2_rate": float(group["is_cic2"].mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _feature_bucket_summary(trades: pd.DataFrame) -> pd.DataFrame:
    features = {
        "beta_extreme_strength": "rank_beta_extreme_strength",
        "local_volume_shock_strength": "rank_local_volume_shock_strength",
        "market_impulse_density": "rank_market_impulse_density",
        "cluster_impulse_density": "rank_cluster_impulse_density",
        "liquidity": "rank_liquidity",
        "reclaim_quality": "rank_reclaim_quality",
    }
    rows = []
    for pool_spec in POOLS:
        pool = _pool_trades(trades, pool_spec.pool)
        pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)].copy()
        if pool.empty:
            continue
        for feature, col in features.items():
            vals = pd.to_numeric(pool.get(col), errors="coerce")
            if vals.notna().sum() < 8 or vals.nunique(dropna=True) < 2:
                continue
            bucket = pd.qcut(vals.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
            for label, group in pool.groupby(bucket, observed=True, sort=False):
                net = pd.to_numeric(group["net_return"], errors="coerce")
                rows.append(
                    {
                        "pool": pool_spec.pool,
                        "feature": feature,
                        "bucket": str(label),
                        "trades": int(len(group)),
                        "net20": float(net.mean()),
                        "good_trade_rate": float((net > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _skipped_attribution(skipped: pd.DataFrame) -> pd.DataFrame:
    if skipped.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["pool", "ranking", "max_positions", "skip_reason"]
    for keys, group in skipped.groupby(group_cols, sort=False, dropna=False):
        net = pd.to_numeric(group["net_return"], errors="coerce")
        rows.append(
            {
                "pool": keys[0],
                "ranking": keys[1],
                "max_positions": keys[2],
                "skip_reason": keys[3],
                "skipped_trades": int(len(group)),
                "skipped_avg_net20": float(net.mean()) if len(net) else np.nan,
                "skipped_good_trade_rate": float((net > 0).mean()) if len(net) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _capacity_curve(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    focus = summary[
        summary["pool"].astype(str).isin(["P0_CIC1_ONLY", "P2_CIC1_CIC2_COMBINED"])
        & summary["ranking"].astype(str).isin(["cluster_impulse_density_high", "first_come_first_served"])
    ].copy()
    if focus.empty:
        return pd.DataFrame()
    focus["cost_label"] = (
        pd.to_numeric(focus["cost_single_side_bps"], errors="coerce").astype("Int64").astype(str) + "bp"
    )
    id_cols = ["pool", "ranking", "max_positions"]
    first_cols = [
        "selected_trades",
        "skipped_trades",
        "max_drawdown_proxy",
        "capital_utilization",
        "avg_holding_minutes",
        "return_per_capital_day",
        "max_month_contribution",
        "max_symbol_contribution",
    ]
    base = (
        focus.sort_values(["pool", "ranking", "max_positions", "cost_single_side_bps"])
        .groupby(id_cols, as_index=False, sort=False)[first_cols]
        .first()
    )
    net = focus.pivot_table(
        index=id_cols,
        columns="cost_label",
        values="net_expectancy",
        aggfunc="first",
        observed=True,
    ).reset_index()
    skipped = focus.pivot_table(
        index=id_cols,
        columns="cost_label",
        values="skipped_avg_net",
        aggfunc="first",
        observed=True,
    ).reset_index()
    skipped = skipped.rename(columns={col: f"skipped_{col}" for col in skipped.columns if col not in id_cols})
    out = base.merge(net, on=id_cols, how="left").merge(skipped, on=id_cols, how="left")
    if "20bp" in out.columns and "skipped_20bp" in out.columns:
        out["selected_minus_skipped_20bp"] = out["20bp"] - out["skipped_20bp"]
    order = {str(pos): idx for idx, pos in enumerate([1, 2, 3, 5, 8, 10, "unlimited"])}
    out["_order"] = out["max_positions"].astype(str).map(order).fillna(999)
    return out.sort_values(["pool", "ranking", "_order"]).drop(columns=["_order"])


def _write_notes(report_root: Path, summary: pd.DataFrame, random_df: pd.DataFrame) -> None:
    lines = [
        "# v0.9B Portfolio Ranking Diagnostic",
        "",
        "Purpose: keep CIC/MIR1 signal definitions frozen and test portfolio ranking under max_positions constraints.",
        "",
        "Cluster features are used only as ranking diagnostics, not as a proven cluster alpha.",
        "",
    ]
    focus = summary[
        pd.to_numeric(summary["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)
        & summary["max_positions"].astype(str).isin(["3", "5"])
    ].copy()
    if not focus.empty:
        best = focus.sort_values("net_expectancy", ascending=False).head(8)
        lines.append("## Best focal net20 rows")
        for row in best.itertuples(index=False):
            lines.append(
                f"- {row.pool} / {row.ranking} / max={row.max_positions}: "
                f"trades={row.selected_trades}, net20={row.net_expectancy:.4%}, "
                f"random_pct={getattr(row, 'percentile_vs_random', np.nan):.2%}."
            )
    if not random_df.empty:
        lines.append("")
        lines.append(f"Random permutations per pool/max_positions: {RANDOM_PERMUTATIONS}.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v09b_portfolio_ranking(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= TOP_N]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trades, signals = _stream_frozen_trades(feature_path, rank30, rank90, regime, config, report_root)
    trades = _prepare_trade_features(trades)
    deterministic, timeline, skipped = _run_deterministic_rankings(trades)
    random_dist = _run_random_rankings(trades)
    ranking_summary = _attach_random_comparison(deterministic, random_dist)
    pool_compare = _pool_comparison(trades)
    feature_buckets = _feature_bucket_summary(trades)
    skipped_attr = _skipped_attribution(skipped)
    max_summary = ranking_summary[
        pd.to_numeric(ranking_summary["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)
    ].copy()
    capacity = _capacity_curve(ranking_summary)
    outputs = {
        "ranking_summary": report_root / "ranking_summary.csv",
        "ranking_random_distribution": report_root / "ranking_random_distribution.csv",
        "max_positions_summary": report_root / "max_positions_summary.csv",
        "portfolio_timeline": report_root / "portfolio_timeline.csv",
        "skipped_trade_attribution": report_root / "skipped_trade_attribution.csv",
        "feature_rank_bucket_summary": report_root / "feature_rank_bucket_summary.csv",
        "cic_mir1_pool_comparison": report_root / "cic_mir1_pool_comparison.csv",
        "capacity_curve": report_root / "capacity_curve.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    ranking_summary.to_csv(outputs["ranking_summary"], index=False)
    random_dist.to_csv(outputs["ranking_random_distribution"], index=False)
    max_summary.to_csv(outputs["max_positions_summary"], index=False)
    timeline.to_csv(outputs["portfolio_timeline"], index=False)
    skipped_attr.to_csv(outputs["skipped_trade_attribution"], index=False)
    feature_buckets.to_csv(outputs["feature_rank_bucket_summary"], index=False)
    pool_compare.to_csv(outputs["cic_mir1_pool_comparison"], index=False)
    capacity.to_csv(outputs["capacity_curve"], index=False)
    signals.to_csv(report_root / "signal_counts.csv", index=False)
    _write_notes(report_root, ranking_summary, random_dist)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "select_portfolio",
    "write_v09b_portfolio_ranking",
]
