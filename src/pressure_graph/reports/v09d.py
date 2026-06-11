from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07b import TOP_N
from pressure_graph.reports.v09b import (
    FOCAL_COST,
    RANKING_SCORE_COLUMNS,
    _max_contribution,
    _month_cap_expectancy,
    _pool_trades,
    _prepare_trade_features,
    _safe_numeric,
    _stream_frozen_trades,
    select_portfolio,
)


REPORT_ROOT = Path("reports/v0_9d_cic_capacity_architecture")
FOCAL_POOLS = ["P0_CIC1_ONLY", "P2_CIC1_CIC2_COMBINED"]
BASKET_MAX_POSITIONS = [5, 8, 10, 15, 20, 10_000]
BURST_WINDOWS = ["1h", "2h", "4h"]
RESERVE_MAX_POSITIONS = [10, 15]
RESERVE_FRACTIONS = [0.30, 0.50]
RESERVE_PHASE_MINUTES = [30, 60]
REPLACEMENT_MAX_POSITIONS = [5, 10]
REPLACEMENT_THRESHOLDS = [0.00, 0.10, 0.20]
CAPITAL_LOCK_HOURS = [2, 4, 8, 12]


@dataclass(frozen=True)
class ArchitectureSelection:
    selected: pd.DataFrame
    skipped: pd.DataFrame


def _max_pos_label(value: int) -> str:
    return "unlimited" if int(value) >= 10_000 else str(int(value))


def _capital_units(selected: pd.DataFrame, max_positions: int) -> int:
    if int(max_positions) < 10_000:
        return max(1, int(max_positions))
    return max(1, _max_concurrent_positions(selected))


def _max_concurrent_positions(trades: pd.DataFrame, exit_col: str = "exit_time") -> int:
    if trades.empty:
        return 0
    events: list[tuple[pd.Timestamp, int]] = []
    for row in trades.itertuples(index=False):
        entry = pd.Timestamp(getattr(row, "entry_time"))
        exit_time = pd.Timestamp(getattr(row, exit_col))
        if pd.isna(entry) or pd.isna(exit_time):
            continue
        events.append((entry, 1))
        events.append((exit_time, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        max_active = max(max_active, active)
    return int(max_active)


def _period_hours(trades: pd.DataFrame, exit_col: str = "exit_time") -> float:
    if trades.empty:
        return np.nan
    start = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce").min()
    end = pd.to_datetime(trades[exit_col], utc=True, errors="coerce").max()
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return max(float((end - start).total_seconds() / 3600.0), 1.0)


def _month_sum(sample: pd.DataFrame, capital_units: int) -> pd.Series:
    if sample.empty:
        return pd.Series(dtype=float)
    return sample.groupby("month", sort=False, dropna=False)["net_return"].sum() / max(1, capital_units)


def _portfolio_arch_metrics(
    selected: pd.DataFrame,
    skipped: pd.DataFrame,
    *,
    architecture: str,
    pool: str,
    rule: str,
    max_positions: int,
    notes: str = "",
) -> dict[str, object]:
    selected = selected.copy()
    skipped = skipped.copy()
    net = pd.to_numeric(selected.get("net_return", pd.Series(dtype=float)), errors="coerce")
    skipped_net = pd.to_numeric(skipped.get("net_return", pd.Series(dtype=float)), errors="coerce")
    capital_units = _capital_units(selected, max_positions)
    contribution = net / max(1, capital_units)
    equity = contribution.cumsum()
    dd = equity - equity.cummax()
    period_hours = _period_hours(selected)
    holding_hours = pd.to_numeric(selected.get("holding_minutes", pd.Series(dtype=float)), errors="coerce").sum() / 60.0
    month_net = _month_sum(selected, capital_units)
    return {
        "architecture": architecture,
        "pool": pool,
        "rule": rule,
        "cost_single_side_bps": FOCAL_COST,
        "max_positions": _max_pos_label(max_positions),
        "capital_units": capital_units,
        "selected_trades": int(len(selected)),
        "skipped_trades": int(len(skipped)),
        "selected_net20": float(net.mean()) if len(net) else np.nan,
        "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "selected_minus_skipped_net20": float(net.mean() - skipped_net.mean()) if len(net) and len(skipped_net) else np.nan,
        "portfolio_net20": float(contribution.sum()) if len(contribution) else 0.0,
        "return_per_capital_day": float(contribution.sum() / period_hours * 24.0) if period_hours else np.nan,
        "return_per_position_hour": float(net.sum() / holding_hours) if holding_hours else np.nan,
        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
        "capital_utilization": float(holding_hours / (period_hours * capital_units)) if period_hours else np.nan,
        "avg_concurrent_positions": float(holding_hours / period_hours) if period_hours else np.nan,
        "max_concurrent_positions": _max_concurrent_positions(selected),
        "worst_month": str(month_net.idxmin()) if len(month_net) else "",
        "worst_month_net": float(month_net.min()) if len(month_net) else np.nan,
        "month_cap35_net20": _month_cap_expectancy(selected),
        "max_month_contribution": _max_contribution(selected, "month"),
        "max_symbol_contribution": _max_contribution(selected, "symbol"),
        "selected_good_trade_rate": float((net > 0).mean()) if len(net) else np.nan,
        "skipped_good_trade_rate": float((skipped_net > 0).mean()) if len(skipped_net) else np.nan,
        "notes": notes,
    }


def _focus_pool(trades: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    pool = _pool_trades(trades, pool_name)
    if pool.empty:
        return pool
    pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)].copy()
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce")
    return pool.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _add_burst_id(trades: pd.DataFrame, window: str) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.sort_values(["entry_time", "symbol"]).copy()
    gap = pd.Timedelta(window)
    burst_ids: list[str] = []
    burst_idx = -1
    last_time: pd.Timestamp | None = None
    for entry in pd.to_datetime(out["entry_time"], utc=True, errors="coerce"):
        if last_time is None or pd.isna(entry) or entry - last_time > gap:
            burst_idx += 1
        burst_ids.append(f"{window}_burst_{burst_idx:04d}")
        if not pd.isna(entry):
            last_time = entry
    out["burst_id"] = burst_ids
    out["burst_window"] = window
    starts = out.groupby("burst_id", sort=False)["entry_time"].transform("min")
    out["minutes_since_burst_start"] = (out["entry_time"] - starts).dt.total_seconds() / 60.0
    out["burst_size"] = out.groupby("burst_id", sort=False)["symbol"].transform("size")
    out["burst_order"] = out.groupby("burst_id", sort=False).cumcount() + 1
    return out


def _weighted_burst_return(group: pd.DataFrame, policy: str) -> float:
    net = pd.to_numeric(group["net_return"], errors="coerce")
    if group.empty:
        return np.nan
    if policy == "liquidity_weighted":
        score = pd.to_numeric(group.get("rank_liquidity"), errors="coerce")
    elif policy == "beta_weighted":
        score = pd.to_numeric(group.get("rank_beta_extreme_strength"), errors="coerce")
    else:
        return float(net.mean())
    score = score.rank(pct=True).fillna(0.5).clip(lower=0.05)
    weights = score / score.sum() if score.sum() else pd.Series(1.0 / len(group), index=group.index)
    return float((net * weights).sum())


def _burst_basket_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policies = {
        "burst_all_equal": None,
        "burst_first_10_equal": 10,
        "burst_first_15_equal": 15,
        "burst_liquidity_weighted": None,
        "burst_beta_weighted": None,
    }
    for pool_name in FOCAL_POOLS:
        pool = _focus_pool(trades, pool_name)
        if pool.empty:
            continue
        for window in BURST_WINDOWS:
            local = _add_burst_id(pool, window)
            for policy, limit in policies.items():
                burst_returns = []
                burst_sizes = []
                burst_drawdowns = []
                selected_counts = []
                for burst_id, group in local.groupby("burst_id", sort=False, dropna=False):
                    sample = group.sort_values(["entry_time", "symbol"]).copy()
                    if limit is not None:
                        sample = sample.head(limit)
                    if sample.empty:
                        continue
                    burst_return = _weighted_burst_return(sample, policy.replace("burst_", ""))
                    burst_returns.append(burst_return)
                    burst_sizes.append(int(group["burst_size"].iloc[0]))
                    selected_counts.append(int(len(sample)))
                    contrib = pd.to_numeric(sample["net_return"], errors="coerce") / max(1, len(sample))
                    equity = contrib.cumsum()
                    burst_drawdowns.append(float((equity - equity.cummax()).min()))
                vals = pd.Series(burst_returns, dtype="float64")
                rows.append(
                    {
                        "pool": pool_name,
                        "burst_window": window,
                        "basket_policy": policy,
                        "bursts": int(len(vals)),
                        "avg_burst_return": float(vals.mean()) if len(vals) else np.nan,
                        "median_burst_return": float(vals.median()) if len(vals) else np.nan,
                        "burst_hit_rate": float((vals > 0).mean()) if len(vals) else np.nan,
                        "worst_burst": float(vals.min()) if len(vals) else np.nan,
                        "avg_burst_size": float(np.mean(burst_sizes)) if burst_sizes else np.nan,
                        "avg_selected_per_burst": float(np.mean(selected_counts)) if selected_counts else np.nan,
                        "worst_burst_drawdown": float(np.min(burst_drawdowns)) if burst_drawdowns else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _basket_capacity_curve(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    timeline_frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    for pool_name in FOCAL_POOLS:
        pool = _focus_pool(trades, pool_name)
        if pool.empty:
            continue
        for max_positions in BASKET_MAX_POSITIONS:
            selected, skipped = select_portfolio(
                pool,
                score_col=RANKING_SCORE_COLUMNS["first_come_first_served"],
                max_positions=max_positions,
            )
            rows.append(
                _portfolio_arch_metrics(
                    selected,
                    skipped,
                    architecture="basket_capacity",
                    pool=pool_name,
                    rule="first_come_equal_weight",
                    max_positions=max_positions,
                    notes="Basket capacity accepts early eligible CIC trades and scales weight by capacity units.",
                )
            )
            if max_positions in {10, 15, 20}:
                if not selected.empty:
                    local = selected.copy()
                    local["architecture"] = "basket_capacity"
                    local["rule"] = "first_come_equal_weight"
                    local["max_positions"] = _max_pos_label(max_positions)
                    timeline_frames.append(local)
                if not skipped.empty:
                    local = skipped.copy()
                    local["architecture"] = "basket_capacity"
                    local["rule"] = "first_come_equal_weight"
                    local["max_positions"] = _max_pos_label(max_positions)
                    skipped_frames.append(local)
    return (
        pd.DataFrame(rows),
        pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _select_with_reserve(
    trades: pd.DataFrame,
    *,
    max_positions: int,
    reserve_fraction: float,
    phase_minutes: int,
    burst_window: str,
) -> ArchitectureSelection:
    if trades.empty:
        return ArchitectureSelection(trades.copy(), trades.copy())
    ranked = _add_burst_id(trades, burst_window).sort_values(["entry_time", "symbol"]).copy()
    active_exits: list[pd.Timestamp] = []
    active_symbols: dict[str, pd.Timestamp] = {}
    selected_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    reserved_slots = int(np.ceil(max_positions * reserve_fraction))
    early_cap = max(1, max_positions - reserved_slots)
    for row in ranked.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active_exits = [exit_time for exit_time in active_exits if exit_time > entry]
        active_symbols = {symbol: exit_time for symbol, exit_time in active_symbols.items() if exit_time > entry}
        payload = row._asdict()
        payload["active_positions_at_decision"] = len(active_exits)
        payload["reserve_fraction"] = reserve_fraction
        payload["phase_minutes"] = phase_minutes
        minute = float(getattr(row, "minutes_since_burst_start", np.nan))
        allowed = early_cap if np.isfinite(minute) and minute <= phase_minutes else max_positions
        payload["allowed_positions_at_decision"] = allowed
        symbol = str(getattr(row, "symbol"))
        if symbol in active_symbols:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if len(active_exits) >= allowed:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "reserved_capacity"
            skipped_rows.append(payload)
            continue
        payload["selection_status"] = "selected"
        payload["skip_reason"] = ""
        selected_rows.append(payload)
        active_exits.append(pd.Timestamp(row.exit_time))
        active_symbols[symbol] = pd.Timestamp(row.exit_time)
    return ArchitectureSelection(pd.DataFrame(selected_rows), pd.DataFrame(skipped_rows))


def _reserve_capacity_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pool_name in FOCAL_POOLS:
        pool = _focus_pool(trades, pool_name)
        if pool.empty:
            continue
        for max_positions in RESERVE_MAX_POSITIONS:
            for reserve_fraction in RESERVE_FRACTIONS:
                for phase in RESERVE_PHASE_MINUTES:
                    selected = _select_with_reserve(
                        pool,
                        max_positions=max_positions,
                        reserve_fraction=reserve_fraction,
                        phase_minutes=phase,
                        burst_window="1h",
                    )
                    rows.append(
                        _portfolio_arch_metrics(
                            selected.selected,
                            selected.skipped,
                            architecture="reserve_capacity",
                            pool=pool_name,
                            rule=f"reserve_{int(reserve_fraction * 100)}pct_phase_{phase}m",
                            max_positions=max_positions,
                            notes="Reserve diagnostic keeps part of capacity unused during early burst phase.",
                        )
                    )
    return pd.DataFrame(rows)


def _select_with_replacement(
    trades: pd.DataFrame,
    *,
    max_positions: int,
    score_col: str,
    threshold: float,
) -> ArchitectureSelection:
    if trades.empty:
        return ArchitectureSelection(trades.copy(), trades.copy())
    ranked = trades.copy().reset_index(drop=True)
    ranked["row_id"] = np.arange(len(ranked))
    raw_score = _safe_numeric(ranked, score_col, 0.0)
    ranked["score_pct"] = raw_score.rank(pct=True).fillna(0.5)
    ranked = ranked.sort_values(["entry_time", "score_pct", "symbol"], ascending=[True, False, True])
    active: list[dict[str, object]] = []
    selected_ids: set[int] = set()
    skipped_rows: list[dict[str, object]] = []
    for row in ranked.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active = [item for item in active if pd.Timestamp(item["exit_time"]) > entry]
        active_symbols = {str(item["symbol"]) for item in active}
        payload = row._asdict()
        payload["active_positions_at_decision"] = len(active)
        payload["replacement_threshold"] = threshold
        symbol = str(getattr(row, "symbol"))
        if symbol in active_symbols:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if len(active) < max_positions:
            selected_ids.add(int(getattr(row, "row_id")))
            active.append(
                {
                    "row_id": int(getattr(row, "row_id")),
                    "symbol": symbol,
                    "exit_time": pd.Timestamp(row.exit_time),
                    "score_pct": float(getattr(row, "score_pct")),
                }
            )
            continue
        worst = min(active, key=lambda item: float(item["score_pct"]))
        if float(getattr(row, "score_pct")) > float(worst["score_pct"]) + threshold:
            selected_ids.discard(int(worst["row_id"]))
            skipped_payload = ranked.loc[ranked["row_id"].eq(int(worst["row_id"]))].iloc[0].to_dict()
            skipped_payload["selection_status"] = "skipped"
            skipped_payload["skip_reason"] = "replaced_out_diagnostic"
            skipped_payload["replacement_threshold"] = threshold
            skipped_rows.append(skipped_payload)
            active = [item for item in active if int(item["row_id"]) != int(worst["row_id"])]
            selected_ids.add(int(getattr(row, "row_id")))
            active.append(
                {
                    "row_id": int(getattr(row, "row_id")),
                    "symbol": symbol,
                    "exit_time": pd.Timestamp(row.exit_time),
                    "score_pct": float(getattr(row, "score_pct")),
                }
            )
        else:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "portfolio_full_no_replacement"
            skipped_rows.append(payload)
    selected = ranked[ranked["row_id"].isin(selected_ids)].copy()
    selected["selection_status"] = "selected"
    selected["skip_reason"] = ""
    return ArchitectureSelection(selected, pd.DataFrame(skipped_rows))


def _replacement_rule_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pool_name in FOCAL_POOLS:
        pool = _focus_pool(trades, pool_name)
        if pool.empty:
            continue
        for max_positions in REPLACEMENT_MAX_POSITIONS:
            for score_name in ["rank_composite_simple", "rank_beta_extreme_strength", "rank_cluster_impulse_density"]:
                if score_name not in pool.columns:
                    continue
                for threshold in REPLACEMENT_THRESHOLDS:
                    selected = _select_with_replacement(
                        pool,
                        max_positions=max_positions,
                        score_col=score_name,
                        threshold=threshold,
                    )
                    rows.append(
                        _portfolio_arch_metrics(
                            selected.selected,
                            selected.skipped,
                            architecture="replacement_rule",
                            pool=pool_name,
                            rule=f"{score_name}_replace_if_margin_{threshold:.2f}",
                            max_positions=max_positions,
                            notes="Replacement is a diagnostic using original exits; it is not executable without intratrade mark-to-market.",
                        )
                    )
    return pd.DataFrame(rows)


def _capital_lock_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pool_name in FOCAL_POOLS:
        pool = _focus_pool(trades, pool_name)
        if pool.empty:
            continue
        for max_positions in [5, 10]:
            for hours in CAPITAL_LOCK_HOURS:
                local = pool.copy()
                local["actual_exit_time"] = local["exit_time"]
                local["exit_time"] = pd.to_datetime(local["entry_time"], utc=True, errors="coerce") + pd.Timedelta(hours=hours)
                actual_exit = pd.to_datetime(local["actual_exit_time"], utc=True, errors="coerce")
                local["exit_time"] = pd.to_datetime(local["exit_time"], utc=True, errors="coerce").where(
                    pd.to_datetime(local["exit_time"], utc=True, errors="coerce") < actual_exit,
                    actual_exit,
                )
                selected, skipped = select_portfolio(
                    local,
                    score_col=RANKING_SCORE_COLUMNS["first_come_first_served"],
                    max_positions=max_positions,
                )
                rows.append(
                    _portfolio_arch_metrics(
                        selected,
                        skipped,
                        architecture="capital_lock_diagnostic",
                        pool=pool_name,
                        rule=f"release_after_{hours}h_for_capacity_only",
                        max_positions=max_positions,
                        notes="Uses original trade returns and shortened exit only for capacity release diagnostics.",
                    )
                )
    return pd.DataFrame(rows)


def _selected_vs_skipped_by_burst(timeline: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not timeline.empty:
        selected = _add_burst_id(timeline, "1h")
        selected["side"] = "selected"
        frames.append(selected)
    if not skipped.empty:
        skipped_local = _add_burst_id(skipped, "1h")
        skipped_local["side"] = "skipped"
        frames.append(skipped_local)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    group_cols = ["architecture", "pool", "rule", "max_positions", "burst_id"]
    for keys, group in combined.groupby(group_cols, sort=False, dropna=False):
        selected_group = group[group["side"].eq("selected")]
        skipped_group = group[group["side"].eq("skipped")]
        selected_net = pd.to_numeric(selected_group.get("net_return", pd.Series(dtype=float)), errors="coerce")
        skipped_net = pd.to_numeric(skipped_group.get("net_return", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "architecture": keys[0],
                "pool": keys[1],
                "rule": keys[2],
                "max_positions": keys[3],
                "burst_id": keys[4],
                "selected_trades": int(len(selected_group)),
                "skipped_trades": int(len(skipped_group)),
                "selected_net20": float(selected_net.mean()) if len(selected_net) else np.nan,
                "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
                "selected_minus_skipped_net20": float(selected_net.mean() - skipped_net.mean())
                if len(selected_net) and len(skipped_net)
                else np.nan,
                "burst_size": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _write_notes(report_root: Path, basket: pd.DataFrame, burst: pd.DataFrame, reserve: pd.DataFrame) -> None:
    lines = [
        "# v0.9D CIC Capacity Architecture",
        "",
        "Purpose: keep CIC/MIR1 alpha definitions frozen and test how the CIC signal pool should be carried at portfolio level.",
        "",
        "No CIC parameters, reclaim thresholds, exit rules, market gates, or beta buckets were optimized in this report.",
        "",
        "Replacement and capital-lock sections are diagnostics. They use original trade outcomes and should not be read as executable PnL until intratrade mark-to-market is added.",
        "",
    ]
    if not basket.empty:
        focus = basket.sort_values("return_per_capital_day", ascending=False).head(6)
        lines.append("## Best Basket Rows")
        for row in focus.itertuples(index=False):
            lines.append(
                f"- {row.pool} max={row.max_positions}: portfolio_net20={row.portfolio_net20:.4%}, "
                f"return_per_capital_day={row.return_per_capital_day:.4%}, selected={row.selected_trades}, skipped={row.skipped_trades}."
            )
        lines.append("")
    if not burst.empty:
        focus = burst.sort_values("avg_burst_return", ascending=False).head(5)
        lines.append("## Best Burst Basket Rows")
        for row in focus.itertuples(index=False):
            lines.append(
                f"- {row.pool} {row.burst_window}/{row.basket_policy}: avg_burst_return={row.avg_burst_return:.4%}, "
                f"bursts={row.bursts}, worst_burst={row.worst_burst:.4%}."
            )
        lines.append("")
    if not reserve.empty:
        focus = reserve.sort_values("return_per_capital_day", ascending=False).head(5)
        lines.append("## Reserve Diagnostics")
        for row in focus.itertuples(index=False):
            lines.append(
                f"- {row.pool} {row.rule} max={row.max_positions}: selected={row.selected_trades}, "
                f"skipped={row.skipped_trades}, portfolio_net20={row.portfolio_net20:.4%}."
            )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _load_capacity_trade_cache(report_root: Path) -> pd.DataFrame:
    cache_path = report_root / "capacity_trade_cache.parquet"
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(cache_path)


def write_v09d_cic_capacity_architecture(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
    *,
    use_cache: bool = True,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    trades = _load_capacity_trade_cache(report_root) if use_cache else pd.DataFrame()
    if trades.empty:
        rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
        symbols = sorted(
            rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= TOP_N]["symbol"]
            .dropna()
            .astype(str)
            .unique()
        )
        regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
        trades, _ = _stream_frozen_trades(feature_path, rank30, rank90, regime, config, report_root)
        trades = _prepare_trade_features(trades)
        if not trades.empty:
            trades.to_parquet(report_root / "capacity_trade_cache.parquet", index=False)
    else:
        trades = _prepare_trade_features(trades)

    basket, timeline, skipped = _basket_capacity_curve(trades)
    burst = _burst_basket_summary(trades)
    reserve = _reserve_capacity_summary(trades)
    replacement = _replacement_rule_summary(trades)
    capital_lock = _capital_lock_summary(trades)
    burst_compare = _selected_vs_skipped_by_burst(timeline, skipped)

    outputs = {
        "basket_capacity_curve": report_root / "basket_capacity_curve.csv",
        "burst_basket_summary": report_root / "burst_basket_summary.csv",
        "reserve_capacity_summary": report_root / "reserve_capacity_summary.csv",
        "replacement_rule_summary": report_root / "replacement_rule_summary.csv",
        "capital_lock_summary": report_root / "capital_lock_summary.csv",
        "selected_vs_skipped_by_burst": report_root / "selected_vs_skipped_by_burst.csv",
        "portfolio_timeline": report_root / "portfolio_timeline.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    basket.to_csv(outputs["basket_capacity_curve"], index=False)
    burst.to_csv(outputs["burst_basket_summary"], index=False)
    reserve.to_csv(outputs["reserve_capacity_summary"], index=False)
    replacement.to_csv(outputs["replacement_rule_summary"], index=False)
    capital_lock.to_csv(outputs["capital_lock_summary"], index=False)
    burst_compare.to_csv(outputs["selected_vs_skipped_by_burst"], index=False)
    timeline.to_csv(outputs["portfolio_timeline"], index=False)
    skipped.to_csv(report_root / "portfolio_skipped_candidates.csv", index=False)
    _write_notes(report_root, basket, burst, reserve)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "write_v09d_cic_capacity_architecture",
]
