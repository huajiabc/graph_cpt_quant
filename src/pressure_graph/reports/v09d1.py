from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07b import TOP_N
from pressure_graph.reports.v07c2 import _month_setup
from pressure_graph.reports.v07d import ExitModel, _simulate_candidate_exit
from pressure_graph.reports.v07d1 import FROZEN_CANDIDATES, _signal_id
from pressure_graph.reports.v09b import (
    FOCAL_COST,
    RANKING_SCORE_COLUMNS,
    _add_cluster_context_if_available,
    _base_signal_id,
    _pool_trades,
    _prepare_trade_features,
    _read_v09a_membership,
    _release_process_memory,
    _safe_numeric,
    select_portfolio,
)
from pressure_graph.reports.v09d import (
    _portfolio_arch_metrics,
    _selected_vs_skipped_by_burst,
)


REPORT_ROOT = Path("reports/v0_9d1_burst_capacity_execution")

EXIT_VARIANTS = [
    ExitModel("E0_vol_regime_fast", "vol_regime_fast", 16),
    ExitModel("E1_true_2h_time_stop", "vol_regime_fast", 8),
    ExitModel("E2_true_4h_time_stop", "vol_regime_fast", 16),
    ExitModel("E3_checkpoint_return_lt_0", "checkpoint_return", 16, partial_tp=0.0),
    ExitModel("E4_checkpoint_return_lt_1pct", "checkpoint_return", 16, partial_tp=0.01),
    ExitModel("E5_checkpoint_mfe_lt_2pct", "checkpoint_mfe", 16, capture_ratio=0.02),
]
CAPACITY_CANDIDATES = [item for item in FROZEN_CANDIDATES if item.candidate in {"CIC1_beta_extreme", "CIC2_beta_broad"}]

TRUE_TIME_STOP_VARIANTS = ["E0_vol_regime_fast", "E1_true_2h_time_stop", "E2_true_4h_time_stop"]
CHECKPOINT_VARIANTS = [
    "E3_checkpoint_return_lt_0",
    "E4_checkpoint_return_lt_1pct",
    "E5_checkpoint_mfe_lt_2pct",
]
FOCAL_POOLS = ["P0_CIC1_ONLY", "P2_CIC1_CIC2_COMBINED"]
MAX_POSITIONS = [5, 8, 10]


def _selected_exit_variants() -> list[ExitModel]:
    requested = {
        item.strip()
        for item in os.environ.get("V09D1_EXIT_VARIANTS", "").split(",")
        if item.strip()
    }
    if not requested:
        return EXIT_VARIANTS
    return [model for model in EXIT_VARIANTS if model.exit_type in requested]


def _selected_months(months: list[pd.Timestamp]) -> list[pd.Timestamp]:
    requested = {
        item.strip()
        for item in os.environ.get("V09D1_MONTHS", "").split(",")
        if item.strip()
    }
    if not requested:
        return months
    out = []
    for month in months:
        label = pd.Timestamp(month).strftime("%Y-%m")
        compact = pd.Timestamp(month).strftime("%Y%m")
        if label in requested or compact in requested:
            out.append(month)
    return out


def _stream_execution_trades(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> pd.DataFrame:
    trade_path = report_root / "_v09d1_trades_tmp.csv"
    if trade_path.exists():
        trade_path.unlink()
    membership = _read_v09a_membership()
    wrote_trades = False
    all_months = sorted(pd.to_datetime(rank30["month_start"], utc=True, errors="coerce").dropna().drop_duplicates().tolist())
    months = _selected_months(all_months)
    exit_variants = _selected_exit_variants()
    for model_idx, model in enumerate(exit_variants, start=1):
        for idx, month_start in enumerate(months, start=1):
            month_start = pd.Timestamp(month_start)
            all_idx = all_months.index(month_start)
            next_month = (
                pd.Timestamp(all_months[all_idx + 1])
                if all_idx + 1 < len(all_months)
                else month_start + pd.DateOffset(months=1)
            )
            sim_window, _, symbols = _month_setup(feature_path, rank30, rank90, regime, config, month_start, next_month)
            if sim_window.empty:
                continue
            sim_window = _add_cluster_context_if_available(sim_window, membership, month_start)
            frames: list[pd.DataFrame] = []
            for candidate in CAPACITY_CANDIDATES:
                trades, _ = _simulate_candidate_exit(sim_window, candidate, model, config)
                if trades.empty:
                    continue
                trades["candidate"] = candidate.candidate
                trades["exit_variant"] = model.exit_type
                frames.append(trades)
            month_trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not month_trades.empty:
                month_trades["signal_id"] = _signal_id(month_trades)
                month_trades["base_signal_id"] = _base_signal_id(month_trades)
                month_trades.to_csv(trade_path, mode="a", header=not wrote_trades, index=False)
                wrote_trades = True
            del sim_window, frames, month_trades
            _release_process_memory()
            print(
                f"v0.9D.1 variant {model_idx}/{len(exit_variants)} {model.exit_type} "
                f"month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}",
                flush=True,
            )
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    if trade_path.exists():
        trade_path.unlink()
    return trades


def _focus_cost(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = _prepare_trade_features(trades)
    out = out[pd.to_numeric(out["cost_single_side_bps"], errors="coerce").eq(FOCAL_COST)].copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    return out.sort_values(["exit_variant", "entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _pool_variant(trades: pd.DataFrame, pool_name: str, exit_variant: str) -> pd.DataFrame:
    sample = trades[trades["exit_variant"].astype(str).eq(exit_variant)].copy()
    if sample.empty:
        return sample
    return _pool_trades(sample, pool_name)


def _portfolio_rows_for_variants(
    trades: pd.DataFrame,
    variants: list[str],
    *,
    architecture: str,
    rule_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    timeline_frames = []
    skipped_frames = []
    for exit_variant in variants:
        for pool_name in FOCAL_POOLS:
            pool = _pool_variant(trades, pool_name, exit_variant)
            if pool.empty:
                continue
            for max_positions in MAX_POSITIONS:
                selected, skipped = select_portfolio(
                    pool,
                    score_col=RANKING_SCORE_COLUMNS["first_come_first_served"],
                    max_positions=max_positions,
                )
                row = _portfolio_arch_metrics(
                    selected,
                    skipped,
                    architecture=architecture,
                    pool=pool_name,
                    rule=f"{rule_prefix}_{exit_variant}",
                    max_positions=max_positions,
                    notes="True exit variant recalculates trade exits before portfolio capacity selection.",
                )
                row["exit_variant"] = exit_variant
                rows.append(row)
                if pool_name == "P2_CIC1_CIC2_COMBINED" and max_positions == 8:
                    for frame, status in [(selected, "selected"), (skipped, "skipped")]:
                        if frame.empty:
                            continue
                        local = frame.copy()
                        local["selection_status"] = status
                        local["exit_variant"] = exit_variant
                        local["architecture"] = architecture
                        local["rule"] = f"{rule_prefix}_{exit_variant}"
                        local["max_positions"] = max_positions
                        if status == "selected":
                            timeline_frames.append(local)
                        else:
                            skipped_frames.append(local)
    return (
        pd.DataFrame(rows),
        pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _shadow_portfolio_spec() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "portfolio_id": "P2_CIC_COMBINED_BASKET_MAX8",
                "pool": "CIC1 + CIC2 combined",
                "max_positions": 8,
                "sizing": "equal_notional",
                "ranking": "first_come_basket",
                "exit": "vol_regime_fast",
                "status": "shadow_only",
                "real_live_allowed": False,
            }
        ]
    )


def _basket_max8_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary[
        summary["pool"].astype(str).eq("P2_CIC1_CIC2_COMBINED")
        & summary["max_positions"].astype(str).eq("8")
        & summary["exit_variant"].astype(str).eq("E0_vol_regime_fast")
    ].copy()


def _select_with_replacement_diagnostic(
    trades: pd.DataFrame,
    *,
    score_col: str,
    max_positions: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    data = trades.copy().reset_index(drop=True)
    data["row_id"] = np.arange(len(data))
    data["score_pct"] = _safe_numeric(data, score_col, 0.0).rank(pct=True).fillna(0.5)
    data = data.sort_values(["entry_time", "score_pct", "symbol"], ascending=[True, False, True])
    active: list[dict[str, object]] = []
    selected_ids: set[int] = set()
    skipped: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active = [item for item in active if pd.Timestamp(item["exit_time"]) > entry]
        active_symbols = {str(item["symbol"]) for item in active}
        payload = row._asdict()
        payload["replacement_threshold"] = threshold
        if str(row.symbol) in active_symbols:
            payload["skip_reason"] = "symbol_already_active"
            skipped.append(payload)
            continue
        if len(active) < max_positions:
            selected_ids.add(int(row.row_id))
            active.append({"row_id": int(row.row_id), "symbol": row.symbol, "exit_time": row.exit_time, "score_pct": row.score_pct})
            continue
        worst = min(active, key=lambda item: float(item["score_pct"]))
        if float(row.score_pct) > float(worst["score_pct"]) + threshold:
            selected_ids.discard(int(worst["row_id"]))
            old = data[data["row_id"].eq(int(worst["row_id"]))].iloc[0].to_dict()
            old["skip_reason"] = "replaced_out_diagnostic"
            old["replacement_threshold"] = threshold
            skipped.append(old)
            active = [item for item in active if int(item["row_id"]) != int(worst["row_id"])]
            selected_ids.add(int(row.row_id))
            active.append({"row_id": int(row.row_id), "symbol": row.symbol, "exit_time": row.exit_time, "score_pct": row.score_pct})
        else:
            payload["skip_reason"] = "portfolio_full_no_replacement"
            skipped.append(payload)
    return data[data["row_id"].isin(selected_ids)].copy(), pd.DataFrame(skipped)


def _replacement_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exit_variant in ["E0_vol_regime_fast", "E1_true_2h_time_stop"]:
        pool = _pool_variant(trades, "P2_CIC1_CIC2_COMBINED", exit_variant)
        if pool.empty:
            continue
        for max_positions in [5, 8]:
            for score_col in ["rank_beta_extreme_strength", "rank_local_volume_shock_strength", "rank_cluster_impulse_density"]:
                if score_col not in pool.columns:
                    continue
                for threshold in [0.0, 0.1, 0.2]:
                    selected, skipped = _select_with_replacement_diagnostic(
                        pool,
                        score_col=score_col,
                        max_positions=max_positions,
                        threshold=threshold,
                    )
                    row = _portfolio_arch_metrics(
                        selected,
                        skipped,
                        architecture="replacement_diagnostic_after_true_exit",
                        pool="P2_CIC1_CIC2_COMBINED",
                        rule=f"{exit_variant}_{score_col}_margin_{threshold:.1f}",
                        max_positions=max_positions,
                        notes="Replacement still uses full trade outcomes; early liquidation mark-to-market is not yet repriced.",
                    )
                    row["exit_variant"] = exit_variant
                    rows.append(row)
    return pd.DataFrame(rows)


def _add_burst(data: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    if data.empty:
        return data.copy()
    out = data.sort_values(["entry_time", "symbol"]).copy()
    gap = pd.Timedelta(window)
    burst_ids = []
    burst_id = -1
    last_entry: pd.Timestamp | None = None
    for entry in pd.to_datetime(out["entry_time"], utc=True, errors="coerce"):
        if last_entry is None or pd.isna(entry) or entry - last_entry > gap:
            burst_id += 1
        burst_ids.append(f"{window}_burst_{burst_id:04d}")
        if not pd.isna(entry):
            last_entry = entry
    out["burst_id"] = burst_ids
    first = out.groupby("burst_id", sort=False)["entry_time"].transform("min")
    out["minutes_since_burst_start"] = (out["entry_time"] - first).dt.total_seconds() / 60.0
    return out


def _burst_delayed_allocation(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exit_variant in ["E0_vol_regime_fast", "E1_true_2h_time_stop"]:
        pool = _pool_variant(trades, "P2_CIC1_CIC2_COMBINED", exit_variant)
        if pool.empty:
            continue
        pool = _add_burst(pool, "1h")
        for delay in [0, 15, 30]:
            selected_frames = []
            skipped_frames = []
            for _, group in pool.groupby("burst_id", sort=False, dropna=False):
                eligible = group[group["minutes_since_burst_start"] >= delay].copy()
                if eligible.empty:
                    skipped_frames.append(group.assign(skip_reason="before_delayed_allocation_window"))
                    continue
                selected, skipped = select_portfolio(
                    eligible,
                    score_col=RANKING_SCORE_COLUMNS["first_come_first_served"],
                    max_positions=8,
                )
                selected_frames.append(selected)
                excluded = group[~group.index.isin(eligible.index)].copy()
                if not excluded.empty:
                    excluded["skip_reason"] = "before_delayed_allocation_window"
                    skipped_frames.append(excluded)
                if not skipped.empty:
                    skipped_frames.append(skipped)
            selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
            skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
            row = _portfolio_arch_metrics(
                selected_all,
                skipped_all,
                architecture="burst_delayed_allocation_diagnostic",
                pool="P2_CIC1_CIC2_COMBINED",
                rule=f"{exit_variant}_delay_{delay}m_select_first_come_max8",
                max_positions=8,
                notes="Delayed allocation uses original trade entry/exit outcomes; delayed entry price is not repriced.",
            )
            row["exit_variant"] = exit_variant
            row["delay_minutes"] = delay
            rows.append(row)
    return pd.DataFrame(rows)


def _capacity_utilization(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    cols = [
        "architecture",
        "pool",
        "rule",
        "exit_variant",
        "max_positions",
        "selected_trades",
        "skipped_trades",
        "portfolio_net20",
        "return_per_capital_day",
        "capital_utilization",
        "max_concurrent_positions",
        "selected_minus_skipped_net20",
    ]
    return summary[[col for col in cols if col in summary.columns]].copy()


def _write_notes(report_root: Path, time_stop: pd.DataFrame, replacement: pd.DataFrame, delayed: pd.DataFrame) -> None:
    lines = [
        "# v0.9D.1 Burst Capacity Execution",
        "",
        "Purpose: keep CIC signal definitions frozen and test whether real exit/capacity changes can carry the CIC basket better.",
        "",
        "P2_CIC_COMBINED_BASKET_MAX8 is a shadow portfolio candidate, not primary and not real-live.",
        "",
        "True time-stop rows recalculate exit price from the 15m path. Replacement and delayed-allocation rows remain diagnostics unless explicitly marked otherwise.",
        "",
    ]
    if not time_stop.empty:
        best = time_stop.sort_values("return_per_capital_day", ascending=False).head(6)
        lines.append("## Best True Time-Stop Rows")
        for row in best.itertuples(index=False):
            lines.append(
                f"- {row.pool} {row.exit_variant} max={row.max_positions}: "
                f"portfolio_net20={row.portfolio_net20:.4%}, return_per_capital_day={row.return_per_capital_day:.4%}, "
                f"selected={row.selected_trades}, skipped={row.skipped_trades}."
            )
        lines.append("")
    if not replacement.empty:
        best = replacement.sort_values("return_per_capital_day", ascending=False).head(4)
        lines.append("## Replacement Diagnostics")
        for row in best.itertuples(index=False):
            lines.append(f"- {row.rule} max={row.max_positions}: portfolio_net20={row.portfolio_net20:.4%}.")
        lines.append("")
    if not delayed.empty:
        best = delayed.sort_values("return_per_capital_day", ascending=False).head(4)
        lines.append("## Burst Delayed Allocation Diagnostics")
        for row in best.itertuples(index=False):
            lines.append(f"- {row.rule}: portfolio_net20={row.portfolio_net20:.4%}.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def _load_cache(report_root: Path) -> pd.DataFrame:
    variant_paths = sorted(report_root.glob("execution_trade_cache_*.parquet"))
    if variant_paths:
        frames = []
        for path in variant_paths:
            frame = pd.read_parquet(path)
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = report_root / "execution_trade_cache.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_variant_caches(report_root: Path, trades: pd.DataFrame) -> None:
    if trades.empty or "exit_variant" not in trades.columns:
        return
    month_suffix = ""
    requested_months = [
        item.strip().replace("-", "")
        for item in os.environ.get("V09D1_MONTHS", "").split(",")
        if item.strip()
    ]
    if requested_months:
        month_suffix = "_" + "_".join(requested_months[:1] + requested_months[-1:])
    for exit_variant, frame in trades.groupby("exit_variant", sort=False):
        safe_name = str(exit_variant).replace("/", "_").replace("\\", "_")
        frame.to_parquet(report_root / f"execution_trade_cache_{safe_name}{month_suffix}.parquet", index=False)


def write_v09d1_burst_capacity_execution(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
    *,
    use_cache: bool = True,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    selected_run = bool(os.environ.get("V09D1_EXIT_VARIANTS", "").strip())
    trades = _load_cache(report_root) if use_cache and not selected_run else pd.DataFrame()
    if trades.empty:
        rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
        symbols = sorted(
            rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= TOP_N]["symbol"]
            .dropna()
            .astype(str)
            .unique()
        )
        regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
        trades = _stream_execution_trades(feature_path, rank30, rank90, regime, config, report_root)
        trades = _focus_cost(trades)
        if not trades.empty:
            _write_variant_caches(report_root, trades)
    else:
        trades = _focus_cost(trades)

    time_stop, timeline, skipped = _portfolio_rows_for_variants(
        trades,
        TRUE_TIME_STOP_VARIANTS,
        architecture="true_time_stop",
        rule_prefix="basket_first_come",
    )
    checkpoint, _, _ = _portfolio_rows_for_variants(
        trades,
        CHECKPOINT_VARIANTS,
        architecture="checkpoint_exit",
        rule_prefix="basket_first_come",
    )
    replacement = _replacement_summary(trades)
    delayed = _burst_delayed_allocation(trades)
    after_execution = _selected_vs_skipped_by_burst(timeline, skipped)
    capacity = _capacity_utilization(pd.concat([time_stop, checkpoint, replacement, delayed], ignore_index=True))
    baseline = _basket_max8_baseline(time_stop)

    outputs = {
        "shadow_portfolio_spec": report_root / "shadow_portfolio_spec.csv",
        "basket_max8_baseline": report_root / "basket_max8_baseline.csv",
        "true_time_stop_summary": report_root / "true_time_stop_summary.csv",
        "checkpoint_exit_summary": report_root / "checkpoint_exit_summary.csv",
        "replacement_rule_summary": report_root / "replacement_rule_summary.csv",
        "burst_delayed_allocation": report_root / "burst_delayed_allocation.csv",
        "selected_vs_skipped_after_execution": report_root / "selected_vs_skipped_after_execution.csv",
        "portfolio_timeline": report_root / "portfolio_timeline.csv",
        "capacity_utilization": report_root / "capacity_utilization.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    _shadow_portfolio_spec().to_csv(outputs["shadow_portfolio_spec"], index=False)
    baseline.to_csv(outputs["basket_max8_baseline"], index=False)
    time_stop.to_csv(outputs["true_time_stop_summary"], index=False)
    checkpoint.to_csv(outputs["checkpoint_exit_summary"], index=False)
    replacement.to_csv(outputs["replacement_rule_summary"], index=False)
    delayed.to_csv(outputs["burst_delayed_allocation"], index=False)
    after_execution.to_csv(outputs["selected_vs_skipped_after_execution"], index=False)
    timeline.to_csv(outputs["portfolio_timeline"], index=False)
    capacity.to_csv(outputs["capacity_utilization"], index=False)
    skipped.to_csv(report_root / "portfolio_skipped_candidates.csv", index=False)
    _write_notes(report_root, time_stop, replacement, delayed)
    return outputs


__all__ = ["REPORT_ROOT", "write_v09d1_burst_capacity_execution"]
