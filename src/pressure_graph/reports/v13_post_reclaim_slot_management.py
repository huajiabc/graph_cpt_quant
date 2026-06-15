"""v1.3 post-reclaim follow-through slot management.

v1.2 found that pre-entry reclaim microstructure does not pass selector-grade
within-burst pairwise tests. This report moves the same orderflow layer to the
slot-management question: after a CIC trade is filled, can the first post-entry
hour identify positions that should release capacity for later CIC candidates?

This is still research-only. It does not change paper-live, shadow selectors,
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
from pressure_graph.reports.v09d import _max_concurrent_positions, _period_hours
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig, _load_or_build_trades
from pressure_graph.reports.v11_orderflow_burst_ranking import EVENT_ORDERFLOW_PATH


REPORT_ROOT = Path("reports/v1_3_post_reclaim_slot_management")
POOL_NAME = "P2_CIC1_CIC2_COMBINED"
MAX_POSITIONS_GRID = (5, 8)
CHECKPOINT_MINUTES = 60
FEATURE_COLUMNS = ("exchange", "symbol", "feature_time", "close")


@dataclass(frozen=True)
class V13Config:
    report_root: Path = REPORT_ROOT
    event_orderflow_path: Path = EVENT_ORDERFLOW_PATH
    checkpoint_minutes: int = CHECKPOINT_MINUTES
    min_post_coverage_ratio: float = 0.5
    v10a: V10AConfig = V10AConfig()


@dataclass(frozen=True)
class CheckpointRule:
    rule_id: str
    description: str


RULES = (
    CheckpointRule("baseline_hold", "No checkpoint early exit; current P2 basket baseline."),
    CheckpointRule("exit_if_checkpoint_net_lte_0", "Exit at 1h if checkpoint net20 is <= 0."),
    CheckpointRule("exit_if_checkpoint_gross_lt_1pct", "Exit at 1h if checkpoint gross return is < +1%."),
    CheckpointRule("exit_if_post_imbalance_lt_0", "Exit at 1h if post-entry 1h buy/sell imbalance is negative."),
    CheckpointRule("exit_if_post_taker_buy_lt_50pct", "Exit at 1h if post-entry 1h taker buy ratio is below 50%."),
    CheckpointRule(
        "exit_if_price_loss_and_orderflow_weak",
        "Exit at 1h if price is not positive and post-entry orderflow is weak.",
    ),
    CheckpointRule(
        "exit_if_price_weak_and_orderflow_weak",
        "Exit at 1h if gross return is < +1% and post-entry orderflow is weak.",
    ),
)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _pool_at_cost(trades: pd.DataFrame, cost: float = FOCAL_COST) -> pd.DataFrame:
    pool = _pool_trades(trades, POOL_NAME)
    if pool.empty:
        return pool.copy()
    pool = pool[pd.to_numeric(pool["cost_single_side_bps"], errors="coerce").eq(float(cost))].copy()
    for col in ("signal_time", "entry_time", "exit_time"):
        pool[col] = pd.to_datetime(pool[col], utc=True, errors="coerce")
    pool["entry_price"] = pd.to_numeric(pool["entry_price"], errors="coerce")
    pool["net_return"] = pd.to_numeric(pool["net_return"], errors="coerce")
    pool["month"] = pool["entry_time"].dt.strftime("%Y-%m")
    return (
        pool.dropna(subset=["signal_time", "entry_time", "exit_time", "entry_price", "net_return"])
        .sort_values(["entry_time", "symbol", "candidate"])
        .reset_index(drop=True)
    )


def _join_orderflow(pool: pd.DataFrame, orderflow: pd.DataFrame, cfg: V13Config) -> pd.DataFrame:
    out = pool.copy()
    if "signal_id" not in out.columns:
        out["signal_id"] = _signal_id(out)
    feature_cols = [
        col
        for col in orderflow.columns
        if col not in {"exchange", "symbol", "candidate", "signal_time", "entry_time"}
    ]
    merged = out.merge(orderflow[feature_cols], on="signal_id", how="left", suffixes=("", "_of"))
    post_cov = pd.to_numeric(merged.get("post_entry_1h_coverage_ratio"), errors="coerce").fillna(0.0)
    merged["post_entry_1h_covered_for_checkpoint"] = post_cov >= cfg.min_post_coverage_ratio
    turnover = _num(merged, "post_entry_1h_turnover").replace(0.0, np.nan)
    merged["post_entry_1h_cvd_intensity"] = _num(merged, "post_entry_1h_cvd_delta_turnover") / turnover
    return merged


def _load_checkpoint_price_frame(feature_path: Path, trades: pd.DataFrame, cfg: V13Config) -> pd.DataFrame:
    symbols = sorted(set(trades["symbol"].dropna().astype(str)))
    start = pd.to_datetime(trades["entry_time"], utc=True).min() - pd.Timedelta("30min")
    end = pd.to_datetime(trades["entry_time"], utc=True).max() + pd.Timedelta(minutes=cfg.checkpoint_minutes + 30)
    prices = pd.read_parquet(feature_path, columns=list(FEATURE_COLUMNS))
    prices["feature_time"] = pd.to_datetime(prices["feature_time"], utc=True, errors="coerce")
    prices = prices[
        prices["symbol"].astype(str).isin(symbols)
        & prices["feature_time"].ge(start)
        & prices["feature_time"].le(end)
    ].copy()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return prices.dropna(subset=["feature_time", "symbol", "close"]).sort_values(["symbol", "feature_time"])


def _attach_checkpoint_prices(pool: pd.DataFrame, prices: pd.DataFrame, cfg: V13Config) -> pd.DataFrame:
    out = pool.copy()
    out["checkpoint_time"] = out["entry_time"] + pd.Timedelta(minutes=cfg.checkpoint_minutes)
    frames = []
    for symbol, group in out.groupby("symbol", sort=False):
        price_group = prices[prices["symbol"].astype(str).eq(str(symbol))].sort_values("feature_time")
        local = group.sort_values("checkpoint_time")
        if price_group.empty:
            local["checkpoint_price_time"] = pd.NaT
            local["checkpoint_price"] = np.nan
            frames.append(local)
            continue
        merged = pd.merge_asof(
            local,
            price_group[["feature_time", "close"]].rename(
                columns={"feature_time": "checkpoint_price_time", "close": "checkpoint_price"}
            ),
            left_on="checkpoint_time",
            right_on="checkpoint_price_time",
            direction="backward",
            tolerance=pd.Timedelta("30min"),
        )
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True) if frames else out
    out["checkpoint_gross_return"] = out["checkpoint_price"] / out["entry_price"] - 1.0
    out["checkpoint_net20"] = out["checkpoint_gross_return"] - 2.0 * FOCAL_COST / 10_000.0
    out["checkpoint_price_covered"] = out["checkpoint_price"].notna()
    return out.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _orderflow_weak(row: pd.Series) -> bool:
    imbalance = pd.to_numeric(pd.Series([row.get("post_entry_1h_buy_sell_imbalance")]), errors="coerce").iloc[0]
    taker = pd.to_numeric(pd.Series([row.get("post_entry_1h_taker_buy_ratio")]), errors="coerce").iloc[0]
    return bool((np.isfinite(imbalance) and imbalance < 0.0) or (np.isfinite(taker) and taker < 0.5))


def _should_checkpoint_exit(row: pd.Series, rule_id: str) -> bool:
    if rule_id == "baseline_hold":
        return False
    if not bool(row.get("post_entry_1h_covered_for_checkpoint", False)):
        return False
    if not bool(row.get("checkpoint_price_covered", False)):
        return False
    checkpoint_time = pd.Timestamp(row["checkpoint_time"])
    original_exit = pd.Timestamp(row["exit_time"])
    if checkpoint_time >= original_exit:
        return False
    checkpoint_net = float(row.get("checkpoint_net20", np.nan))
    checkpoint_gross = float(row.get("checkpoint_gross_return", np.nan))
    imbalance = float(row.get("post_entry_1h_buy_sell_imbalance", np.nan))
    taker = float(row.get("post_entry_1h_taker_buy_ratio", np.nan))
    weak_of = _orderflow_weak(row)
    if rule_id == "exit_if_checkpoint_net_lte_0":
        return np.isfinite(checkpoint_net) and checkpoint_net <= 0.0
    if rule_id == "exit_if_checkpoint_gross_lt_1pct":
        return np.isfinite(checkpoint_gross) and checkpoint_gross < 0.01
    if rule_id == "exit_if_post_imbalance_lt_0":
        return np.isfinite(imbalance) and imbalance < 0.0
    if rule_id == "exit_if_post_taker_buy_lt_50pct":
        return np.isfinite(taker) and taker < 0.5
    if rule_id == "exit_if_price_loss_and_orderflow_weak":
        return np.isfinite(checkpoint_net) and checkpoint_net <= 0.0 and weak_of
    if rule_id == "exit_if_price_weak_and_orderflow_weak":
        return np.isfinite(checkpoint_gross) and checkpoint_gross < 0.01 and weak_of
    raise KeyError(rule_id)


def _apply_checkpoint_rule(trades: pd.DataFrame, rule_id: str) -> pd.DataFrame:
    out = trades.copy()
    early = out.apply(lambda row: _should_checkpoint_exit(row, rule_id), axis=1)
    out["checkpoint_rule"] = rule_id
    out["checkpoint_early_exit"] = early
    out["effective_exit_time"] = out["exit_time"]
    out["effective_net_return"] = out["net_return"]
    out.loc[early, "effective_exit_time"] = out.loc[early, "checkpoint_time"]
    out.loc[early, "effective_net_return"] = out.loc[early, "checkpoint_net20"]
    out["effective_holding_minutes"] = (
        pd.to_datetime(out["effective_exit_time"], utc=True, errors="coerce")
        - pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    return out


def _select_with_effective_exits(trades: pd.DataFrame, max_positions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = trades.sort_values(["entry_time", "rank_first_come_first_served", "symbol"]).copy()
    active: list[tuple[pd.Timestamp, str]] = []
    selected = []
    skipped = []
    for row in ranked.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active = [(exit_time, symbol) for exit_time, symbol in active if exit_time > entry]
        active_symbols = {symbol for _, symbol in active}
        payload = row._asdict()
        payload["active_positions_at_decision"] = len(active)
        if str(row.symbol) in active_symbols:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "symbol_already_active"
            skipped.append(payload)
            continue
        if len(active) >= max_positions:
            payload["selection_status"] = "skipped"
            payload["skip_reason"] = "portfolio_full"
            skipped.append(payload)
            continue
        payload["selection_status"] = "selected"
        payload["skip_reason"] = ""
        selected.append(payload)
        active.append((pd.Timestamp(row.effective_exit_time), str(row.symbol)))
    return pd.DataFrame(selected), pd.DataFrame(skipped)


def _portfolio_row(
    selected: pd.DataFrame,
    skipped: pd.DataFrame,
    rule: CheckpointRule,
    max_positions: int,
    baseline_selected_ids: set[str] | None = None,
) -> dict[str, object]:
    selected_net = _num(selected, "effective_net_return")
    selected_original_net = _num(selected, "net_return")
    skipped_net = _num(skipped, "net_return")
    effective_selected = selected.copy()
    if "effective_exit_time" in effective_selected.columns:
        effective_selected["exit_time"] = effective_selected["effective_exit_time"]
    if "effective_holding_minutes" in effective_selected.columns:
        effective_selected["holding_minutes"] = effective_selected["effective_holding_minutes"]
    period_hours = _period_hours(effective_selected)
    holding_hours = _num(selected, "effective_holding_minutes").sum() / 60.0
    contribution = selected_net / max_positions
    equity = contribution.cumsum()
    dd = equity - equity.cummax()
    new_selected = np.nan
    if baseline_selected_ids is not None and "signal_id" in selected.columns:
        new_selected = int((~selected["signal_id"].astype(str).isin(baseline_selected_ids)).sum())
    return {
        "rule_id": rule.rule_id,
        "description": rule.description,
        "max_positions": max_positions,
        "selected_trades": int(len(selected)),
        "skipped_trades": int(len(skipped)),
        "early_exit_trades": int(selected.get("checkpoint_early_exit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "newly_selected_vs_baseline": new_selected,
        "selected_effective_net20": float(selected_net.mean()) if len(selected_net) else np.nan,
        "selected_original_net20": float(selected_original_net.mean()) if len(selected_original_net) else np.nan,
        "skipped_counterfactual_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "selected_minus_skipped_net20": float(selected_net.mean() - skipped_net.mean()) if len(selected_net) and len(skipped_net) else np.nan,
        "portfolio_net20": float(contribution.sum()) if len(contribution) else 0.0,
        "return_per_capital_day": float(contribution.sum() / period_hours * 24.0) if period_hours else np.nan,
        "capital_utilization": float(holding_hours / (period_hours * max_positions)) if period_hours else np.nan,
        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
        "max_concurrent_positions": _max_concurrent_positions(effective_selected),
        "month_cap35_net20": _month_cap_expectancy(selected.assign(net_return=selected_net)),
    }


def _bucket_summary(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = (
        "checkpoint_net20",
        "post_entry_1h_buy_sell_imbalance",
        "post_entry_1h_taker_buy_ratio",
        "post_entry_1h_cvd_intensity",
    )
    for feature in features:
        values = _num(sample, feature)
        net = _num(sample, "net_return")
        mask = values.notna() & net.notna()
        if int(mask.sum()) < 25:
            continue
        try:
            buckets = pd.qcut(values[mask], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        local = sample.loc[mask].copy()
        local["_bucket"] = buckets
        for bucket, group in local.groupby("_bucket", sort=True):
            rows.append(
                {
                    "feature": feature,
                    "bucket": f"q{int(bucket) + 1}",
                    "trades": int(len(group)),
                    "future_net20_avg": float(_num(group, "net_return").mean()),
                    "checkpoint_net20_avg": float(_num(group, "checkpoint_net20").mean()),
                    "hit_rate": float((_num(group, "net_return") > 0).mean()),
                    "month_cap35_net20": _month_cap_expectancy(group),
                }
            )
    return pd.DataFrame(rows)


def _build_reports(sample: pd.DataFrame, cfg: V13Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    ledgers = []
    skipped_frames = []
    baseline_ids: dict[int, set[str]] = {}
    for max_positions in MAX_POSITIONS_GRID:
        baseline = _apply_checkpoint_rule(sample, "baseline_hold")
        selected, skipped = _select_with_effective_exits(baseline, max_positions)
        baseline_ids[max_positions] = set(selected.get("signal_id", pd.Series(dtype=str)).astype(str))
        summary_rows.append(_portfolio_row(selected, skipped, RULES[0], max_positions))
        ledgers.append(selected.assign(max_positions=max_positions))
        skipped_frames.append(skipped.assign(max_positions=max_positions))
        for rule in RULES[1:]:
            ruled = _apply_checkpoint_rule(sample, rule.rule_id)
            selected, skipped = _select_with_effective_exits(ruled, max_positions)
            summary_rows.append(_portfolio_row(selected, skipped, rule, max_positions, baseline_ids[max_positions]))
            ledgers.append(selected.assign(max_positions=max_positions))
            skipped_frames.append(skipped.assign(max_positions=max_positions))
    summary = pd.DataFrame(summary_rows)
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    skipped = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    buckets = _bucket_summary(sample)
    return summary, ledger, skipped, buckets


def _write_notes(report_root: Path, summary: pd.DataFrame, sample: pd.DataFrame) -> None:
    lines = [
        "# v1.3 Post-Reclaim Follow-Through Slot Management",
        "",
        "Purpose: test whether post-entry 1h follow-through can release weak slots and admit later CIC candidates.",
        "Status: research-only. No paper-live, shadow selector, or real-live permission change.",
        "",
        "## Coverage",
        f"- P2 trades: {len(sample)}",
        f"- checkpoint price covered: {sample['checkpoint_price_covered'].mean():.1%}",
        f"- post-entry 1h orderflow covered: {sample['post_entry_1h_covered_for_checkpoint'].mean():.1%}",
        "",
        "## Best Rules",
    ]
    if summary.empty:
        lines.append("- No summary rows.")
    else:
        for row in summary.sort_values("portfolio_net20", ascending=False).head(6).itertuples(index=False):
            lines.append(
                f"- {row.rule_id} max{row.max_positions}: portfolio_net20={row.portfolio_net20:.4%}, "
                f"early_exits={row.early_exit_trades}, newly_selected={row.newly_selected_vs_baseline}."
            )
    lines.extend(
        [
            "",
            "## Discipline",
            "- Checkpoint rules can only use information available after the 1h checkpoint.",
            "- Early exits are repriced with 15m close as-of checkpoint time and include 20bp round-trip cost.",
            "- A rule must beat baseline after costs and improve selected-vs-skipped before any shadow promotion.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v13_post_reclaim_slot_management(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V13Config = V13Config(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    trades = _load_or_build_trades(feature_path, instruments, config, report_root, cfg.v10a)
    pool = _pool_at_cost(trades, FOCAL_COST)
    prices = _load_checkpoint_price_frame(feature_path, pool, cfg)
    return write_v13_post_reclaim_slot_management_from_trades(pool, prices, cfg)


def write_v13_post_reclaim_slot_management_from_trades(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V13Config = V13Config(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    if not cfg.event_orderflow_path.exists():
        raise FileNotFoundError(
            f"event orderflow not found: {cfg.event_orderflow_path} (run collect-orderflow-history first)"
        )
    pool = _pool_at_cost(trades, FOCAL_COST) if "pool" not in trades.columns else trades.copy()
    if pool.empty:
        raise ValueError("No P2 CIC trades available for v1.3 slot-management report.")
    orderflow = read_parquet(cfg.event_orderflow_path)
    sample = _attach_checkpoint_prices(_join_orderflow(pool, orderflow, cfg), prices, cfg)
    summary, ledger, skipped, buckets = _build_reports(sample, cfg)

    outputs = {
        "checkpoint_rule_summary": report_root / "checkpoint_rule_summary.csv",
        "checkpoint_trade_ledger": report_root / "checkpoint_trade_ledger.csv",
        "checkpoint_skipped_candidates": report_root / "checkpoint_skipped_candidates.csv",
        "post_followthrough_bucket_summary": report_root / "post_followthrough_bucket_summary.csv",
        "checkpoint_enriched_sample": report_root / "checkpoint_enriched_sample.parquet",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["checkpoint_rule_summary"], index=False)
    ledger.to_csv(outputs["checkpoint_trade_ledger"], index=False)
    skipped.to_csv(outputs["checkpoint_skipped_candidates"], index=False)
    buckets.to_csv(outputs["post_followthrough_bucket_summary"], index=False)
    sample.to_parquet(outputs["checkpoint_enriched_sample"], index=False)
    _write_notes(report_root, summary, sample)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V13Config",
    "write_v13_post_reclaim_slot_management",
    "write_v13_post_reclaim_slot_management_from_trades",
]
