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
    _max_contribution,
    _month_cap_expectancy,
    _pool_trades,
    _prepare_trade_features,
    _stream_frozen_trades,
    select_portfolio,
)
from pressure_graph.reports.v09d import (
    REPORT_ROOT as V09D_REPORT_ROOT,
    _add_burst_id,
    _capital_units,
    _max_concurrent_positions,
    _max_pos_label,
    _period_hours,
)


REPORT_ROOT = Path("reports/v1_0_cic_basket_portfolio")
FOCUS_POOLS = ("P0_CIC1_ONLY", "P2_CIC1_CIC2_COMBINED")
CAPACITY_MAX_POSITIONS = (3, 5, 8, 10, 12, 15, 20, 10_000)
BURST_WINDOWS = ("1h", "2h", "4h")
BURST_CAPS = (8, 10, 15, 10_000)
CIC_WEIGHT_SCHEMES = {
    "cic_1_to_1": (1.0, 1.0),
    "cic_2_to_1": (2.0, 1.0),
    "cic_3_to_1": (3.0, 1.0),
}


@dataclass(frozen=True)
class V10AConfig:
    report_root: Path = REPORT_ROOT
    v09d_root: Path = V09D_REPORT_ROOT
    use_cache: bool = True


def _load_capacity_trade_cache(v09d_root: Path) -> pd.DataFrame:
    path = v09d_root / "capacity_trade_cache.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _focus_pool(trades: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    pool = _pool_trades(trades, pool_name)
    if pool.empty:
        return pool.copy()
    pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)].copy()
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce")
    pool["net_return"] = pd.to_numeric(pool["net_return"], errors="coerce")
    return pool.dropna(subset=["entry_time", "exit_time", "net_return"]).sort_values(
        ["entry_time", "symbol", "candidate"]
    ).reset_index(drop=True)


def _month_cap_for_returns(frame: pd.DataFrame, return_col: str, cap: float = 0.35) -> float:
    if frame.empty:
        return np.nan
    total = pd.to_numeric(frame[return_col], errors="coerce").sum()
    cap_value = total * cap if total > 0 else 0.0
    capped = []
    for _, group in frame.groupby("month", sort=False, dropna=False):
        value = pd.to_numeric(group[return_col], errors="coerce").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped) / len(frame)) if len(frame) else np.nan


def _portfolio_metrics(
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
        "capital_utilization": float(holding_hours / (period_hours * capital_units)) if period_hours else np.nan,
        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
        "avg_concurrent_positions": float(holding_hours / period_hours) if period_hours else np.nan,
        "max_concurrent_positions": _max_concurrent_positions(selected),
        "month_cap35_net20": _month_cap_expectancy(selected),
        "max_month_contribution": _max_contribution(selected, "month"),
        "max_symbol_contribution": _max_contribution(selected, "symbol"),
        "notes": notes,
    }


def _capacity_curve(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    timeline_frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    for pool_name in FOCUS_POOLS:
        pool = _focus_pool(trades, pool_name)
        if pool.empty:
            continue
        for max_positions in CAPACITY_MAX_POSITIONS:
            selected, skipped = select_portfolio(pool, score_col="rank_first_come_first_served", max_positions=max_positions)
            rows.append(
                _portfolio_metrics(
                    selected,
                    skipped,
                    architecture="position_capacity_curve",
                    pool=pool_name,
                    rule="first_come_equal_notional",
                    max_positions=max_positions,
                    notes="Position-count capacity with equal notional per open slot.",
                )
            )
            if max_positions in {5, 8, 10, 12, 15, 20}:
                if not selected.empty:
                    timeline_frames.append(
                        selected.assign(
                            architecture="position_capacity_curve",
                            pool=pool_name,
                            rule="first_come_equal_notional",
                            max_positions=_max_pos_label(max_positions),
                        )
                    )
                if not skipped.empty:
                    skipped_frames.append(
                        skipped.assign(
                            architecture="position_capacity_curve",
                            pool=pool_name,
                            rule="first_come_equal_notional",
                            max_positions=_max_pos_label(max_positions),
                        )
                    )
    return (
        pd.DataFrame(rows),
        pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _candidate_weight(candidate: pd.Series, cic1_weight: float, cic2_weight: float) -> pd.Series:
    text = candidate.astype(str)
    return np.select(
        [text.eq("CIC1_beta_extreme"), text.eq("CIC2_beta_broad")],
        [cic1_weight, cic2_weight],
        default=1.0,
    )


def _burst_weighted_return(sample: pd.DataFrame, cic1_weight: float = 1.0, cic2_weight: float = 1.0) -> float:
    if sample.empty:
        return np.nan
    net = pd.to_numeric(sample["net_return"], errors="coerce")
    weights = pd.Series(_candidate_weight(sample["candidate"], cic1_weight, cic2_weight), index=sample.index, dtype="float64")
    weights = weights / weights.sum() if weights.sum() else pd.Series(1.0 / len(sample), index=sample.index)
    return float((net * weights).sum())


def _burst_rows_for_policy(
    pool: pd.DataFrame,
    *,
    pool_name: str,
    burst_window: str,
    cap: int,
    rule: str,
    cic1_weight: float = 1.0,
    cic2_weight: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    local = _add_burst_id(pool, burst_window)
    rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for burst_id, group in local.groupby("burst_id", sort=False, dropna=False):
        selected = group.sort_values(["entry_time", "symbol"]).copy()
        if cap < 10_000:
            selected = selected.head(cap)
        if selected.empty:
            continue
        burst_return = _burst_weighted_return(selected, cic1_weight, cic2_weight)
        contribution = pd.to_numeric(selected["net_return"], errors="coerce") / max(1, len(selected))
        equity = contribution.cumsum()
        burst_start = pd.to_datetime(group["entry_time"], utc=True, errors="coerce").min()
        month = pd.Timestamp(burst_start).strftime("%Y-%m") if pd.notna(burst_start) else ""
        rows.append(
            {
                "architecture": "burst_fixed_budget",
                "pool": pool_name,
                "rule": rule,
                "burst_window": burst_window,
                "max_per_burst": _max_pos_label(cap),
                "burst_id": burst_id,
                "month": month,
                "burst_size": int(len(group)),
                "selected_trades": int(len(selected)),
                "cic1_trades": int(selected["candidate"].astype(str).eq("CIC1_beta_extreme").sum()),
                "cic2_trades": int(selected["candidate"].astype(str).eq("CIC2_beta_broad").sum()),
                "burst_return_net20": burst_return,
                "avg_trade_net20": float(pd.to_numeric(selected["net_return"], errors="coerce").mean()),
                "worst_trade_net20": float(pd.to_numeric(selected["net_return"], errors="coerce").min()),
                "burst_drawdown_proxy": float((equity - equity.cummax()).min()),
                "burst_start": burst_start,
                "burst_end": pd.to_datetime(group["exit_time"], utc=True, errors="coerce").max(),
            }
        )
        detail_frames.append(
            selected.assign(
                architecture="burst_fixed_budget",
                pool=pool_name,
                rule=rule,
                burst_window=burst_window,
                max_per_burst=_max_pos_label(cap),
                burst_id=burst_id,
            )
        )
    return rows, pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()


def _burst_basket_summary(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    burst_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for pool_name in FOCUS_POOLS:
        pool = _focus_pool(trades, pool_name)
        if pool.empty:
            continue
        for window in BURST_WINDOWS:
            for cap in BURST_CAPS:
                rows, detail = _burst_rows_for_policy(
                    pool,
                    pool_name=pool_name,
                    burst_window=window,
                    cap=cap,
                    rule="equal_weight_fixed_burst_budget",
                )
                burst_rows.extend(rows)
                if not detail.empty:
                    detail_frames.append(detail)
            if pool_name == "P2_CIC1_CIC2_COMBINED":
                for scheme, weights in CIC_WEIGHT_SCHEMES.items():
                    rows, detail = _burst_rows_for_policy(
                        pool,
                        pool_name=pool_name,
                        burst_window=window,
                        cap=10_000,
                        rule=f"{scheme}_fixed_burst_budget",
                        cic1_weight=weights[0],
                        cic2_weight=weights[1],
                    )
                    burst_rows.extend(rows)
                    if not detail.empty:
                        detail_frames.append(detail)
    burst_detail = pd.DataFrame(burst_rows)
    if burst_detail.empty:
        return pd.DataFrame(), burst_detail, pd.DataFrame()
    summary_rows = []
    group_cols = ["architecture", "pool", "rule", "burst_window", "max_per_burst"]
    for keys, group in burst_detail.groupby(group_cols, sort=False, dropna=False):
        returns = pd.to_numeric(group["burst_return_net20"], errors="coerce")
        summary_rows.append(
            {
                "architecture": keys[0],
                "pool": keys[1],
                "rule": keys[2],
                "burst_window": keys[3],
                "max_per_burst": keys[4],
                "bursts": int(len(group)),
                "avg_burst_return_net20": float(returns.mean()),
                "median_burst_return_net20": float(returns.median()),
                "portfolio_burst_net20": float(returns.sum()),
                "burst_hit_rate": float((returns > 0).mean()),
                "worst_burst_net20": float(returns.min()),
                "avg_burst_size": float(pd.to_numeric(group["burst_size"], errors="coerce").mean()),
                "avg_selected_per_burst": float(pd.to_numeric(group["selected_trades"], errors="coerce").mean()),
                "worst_burst_drawdown_proxy": float(pd.to_numeric(group["burst_drawdown_proxy"], errors="coerce").min()),
                "month_cap35_burst_net20": _month_cap_for_returns(group, "burst_return_net20"),
                "max_month_contribution": _max_contribution(group.rename(columns={"burst_return_net20": "net_return"}), "month"),
            }
        )
    return (
        pd.DataFrame(summary_rows),
        burst_detail,
        pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(),
    )


def _selected_vs_skipped_by_capacity(timeline: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    if timeline.empty and skipped.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["architecture", "pool", "rule", "max_positions"]
    selected_groups = timeline.groupby(group_cols, sort=False, dropna=False) if not timeline.empty else []
    for keys, selected in selected_groups:
        local_skipped = skipped
        for col, key in zip(group_cols, keys, strict=True):
            if col in local_skipped.columns:
                local_skipped = local_skipped[local_skipped[col].astype(str).eq(str(key))]
        rows.append(
            {
                "architecture": keys[0],
                "pool": keys[1],
                "rule": keys[2],
                "max_positions": keys[3],
                "selected_trades": int(len(selected)),
                "skipped_trades": int(len(local_skipped)),
                "selected_net20": float(pd.to_numeric(selected["net_return"], errors="coerce").mean()),
                "skipped_net20": float(pd.to_numeric(local_skipped.get("net_return", pd.Series(dtype=float)), errors="coerce").mean())
                if len(local_skipped)
                else np.nan,
                "selected_minus_skipped_net20": float(
                    pd.to_numeric(selected["net_return"], errors="coerce").mean()
                    - pd.to_numeric(local_skipped.get("net_return", pd.Series(dtype=float)), errors="coerce").mean()
                )
                if len(local_skipped)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    report_root: Path,
    capacity: pd.DataFrame,
    burst_summary: pd.DataFrame,
    selected_vs_skipped: pd.DataFrame,
) -> None:
    lines = [
        "# v1.0 CIC Basket Portfolio",
        "",
        "Purpose: treat CIC-filtered MIR1 as a basket / burst alpha instead of forcing top-k ranking.",
        "This report does not change paper-live or real-live permissions.",
        "",
    ]
    p2_capacity = capacity[capacity["pool"].eq("P2_CIC1_CIC2_COMBINED")].copy()
    p2_burst = burst_summary[burst_summary["pool"].eq("P2_CIC1_CIC2_COMBINED")].copy()
    if not p2_capacity.empty:
        best_capacity = p2_capacity.sort_values("portfolio_net20", ascending=False).head(1).iloc[0]
        lines.extend(
            [
                "## Decision",
                f"- Current best historical slot-capacity basket: P2 max={best_capacity.max_positions}, "
                f"portfolio_net20={best_capacity.portfolio_net20:.4%}.",
                "- Fixed burst-budget baskets are not promoted; they were negative in this replay.",
                "- Selected trades still trail skipped counterfactuals, so capacity/ranking remains unresolved.",
                "- No paper-live primary or real-live permission changes are made by this report.",
                "",
            ]
        )
    if not p2_capacity.empty:
        lines.append("## Position Capacity")
        focus = p2_capacity[p2_capacity["max_positions"].astype(str).isin(["3", "5", "8", "10", "12", "15", "20"])]
        for row in focus.sort_values("portfolio_net20", ascending=False).head(8).itertuples(index=False):
            lines.append(
                f"- max={row.max_positions}: portfolio_net20={row.portfolio_net20:.4%}, "
                f"selected={row.selected_trades}, skipped={row.skipped_trades}, "
                f"selected-skipped={row.selected_minus_skipped_net20:.4%}."
            )
        lines.append("")
    if not p2_burst.empty:
        lines.append("## Burst Fixed-Budget")
        for row in p2_burst.sort_values("portfolio_burst_net20", ascending=False).head(8).itertuples(index=False):
            lines.append(
                f"- {row.burst_window}/{row.rule}/cap={row.max_per_burst}: "
                f"portfolio_burst_net20={row.portfolio_burst_net20:.4%}, "
                f"avg_burst={row.avg_burst_return_net20:.4%}, worst_burst={row.worst_burst_net20:.4%}."
            )
        lines.append("")
    if not selected_vs_skipped.empty:
        p2_gap = selected_vs_skipped[
            selected_vs_skipped["pool"].eq("P2_CIC1_CIC2_COMBINED")
            & selected_vs_skipped["max_positions"].astype(str).isin(["5", "8", "10", "12", "15", "20"])
        ]
        if not p2_gap.empty:
            best_gap = p2_gap.sort_values("selected_minus_skipped_net20", ascending=False).head(1).iloc[0]
            lines.extend(
                [
                    "## Capacity Interpretation",
                    f"- Best selected-vs-skipped gap among monitored P2 capacities: max={best_gap.max_positions}, "
                    f"delta={best_gap.selected_minus_skipped_net20:.4%}.",
                    "- If selected remains below skipped, the pool should be carried as a wider basket rather than a narrow top-k selection.",
                    "",
                ]
            )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _load_or_build_trades(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
    cfg: V10AConfig,
) -> pd.DataFrame:
    trades = _load_capacity_trade_cache(cfg.v09d_root) if cfg.use_cache else pd.DataFrame()
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
    return _prepare_trade_features(trades)


def write_v10a_cic_basket_portfolio(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V10AConfig = V10AConfig(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    trades = _load_or_build_trades(feature_path, instruments, config, report_root, cfg)

    capacity, timeline, skipped = _capacity_curve(trades)
    burst_summary, burst_detail, burst_trades = _burst_basket_summary(trades)
    selected_vs_skipped = _selected_vs_skipped_by_capacity(timeline, skipped)

    outputs = {
        "capacity_curve": report_root / "capacity_curve.csv",
        "selected_vs_skipped": report_root / "selected_vs_skipped.csv",
        "portfolio_timeline": report_root / "portfolio_timeline.csv",
        "portfolio_skipped_candidates": report_root / "portfolio_skipped_candidates.csv",
        "burst_basket_summary": report_root / "burst_basket_summary.csv",
        "burst_detail": report_root / "burst_detail.csv",
        "burst_trade_detail": report_root / "burst_trade_detail.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    capacity.to_csv(outputs["capacity_curve"], index=False)
    selected_vs_skipped.to_csv(outputs["selected_vs_skipped"], index=False)
    timeline.to_csv(outputs["portfolio_timeline"], index=False)
    skipped.to_csv(outputs["portfolio_skipped_candidates"], index=False)
    burst_summary.to_csv(outputs["burst_basket_summary"], index=False)
    burst_detail.to_csv(outputs["burst_detail"], index=False)
    burst_trades.to_csv(outputs["burst_trade_detail"], index=False)
    _write_notes(report_root, capacity, burst_summary, selected_vs_skipped)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V10AConfig",
    "write_v10a_cic_basket_portfolio",
    "_burst_basket_summary",
    "_capacity_curve",
]
