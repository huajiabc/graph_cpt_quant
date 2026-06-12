from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.config.v07a2 import V07A2Config
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.reports.v09a import REPORT_ROOT as V09A_REPORT_ROOT
from pressure_graph.reports.v09a import add_cluster_graph_features
from pressure_graph.paper_live.v07a2 import (
    _append_decision_log,
    _daily_summary,
    _data_stale,
    _latest_feature_time,
    _latest_market_state,
    _market_gate_audit,
    _primary_portfolio_trades,
    _safe_float,
    _sample_status,
    _shadow_baselines_live,
    _summary,
    add_v07a2_live_columns,
    build_v07a2_paper_ledger,
    prepare_v07a2_features_from_history,
)


REPORT_ROOT = Path("reports/v0_7d2_cic_mir1_paper_live")
PAPER_DATA_ROOT = Path("data/paper/v0_7d2")
OVERFLOW_PORTFOLIO_ID = "P2_MAX8_PLUS_O6_LATE_BURST_OVERFLOW"
SHADOW_PORTFOLIOS = [
    {
        "portfolio_id": "P0_CIC1_CLUSTER_RANK_MAX5",
        "pool": "CIC1_ONLY",
        "description": "CIC1-filtered MIR1 only, ranked by cluster impulse density, max_positions=5.",
        "candidates": ["CIC1_FILTERED_MIR1"],
        "max_positions": 5,
        "score_col": "cluster_impulse_density_at_entry",
    },
    {
        "portfolio_id": "P2_CIC_COMBINED_CLUSTER_RANK_MAX5",
        "pool": "CIC1_CIC2_COMBINED",
        "description": "CIC1 + CIC2 combined, CIC1 priority on duplicate signals, ranked by cluster impulse density, max_positions=5.",
        "candidates": ["CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"],
        "max_positions": 5,
        "score_col": "cluster_impulse_density_at_entry",
        "ranking": "cluster_impulse_density_high",
    },
    {
        "portfolio_id": "P2_CIC_COMBINED_BASKET_MAX8",
        "pool": "CIC1_CIC2_COMBINED",
        "description": "CIC1 + CIC2 combined basket, first-come equal-notional shadow, max_positions=8.",
        "candidates": ["CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"],
        "max_positions": 8,
        "score_col": "__first_come__",
        "ranking": "first_come_basket",
    },
    {
        "portfolio_id": OVERFLOW_PORTFOLIO_ID,
        "pool": "CIC1_CIC2_COMBINED",
        "description": (
            "P2 max8 core basket plus O6 late-burst overflow: "
            "portfolio_full, burst_count_so_far>=9, max_overflow_slots=4, "
            "CIC1 size=0.50 and CIC2 size=0.25."
        ),
        "candidates": ["CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"],
        "max_positions": 8,
        "core_max_positions": 8,
        "score_col": "__first_come__",
        "ranking": "first_come_late_burst_overflow",
        "selection": "late_burst_overflow",
        "overflow_min_burst_count": 9,
        "overflow_max_slots": 4,
        "overflow_size_by_candidate": {"CIC1_FILTERED_MIR1": 0.50, "CIC2_FILTERED_MIR1": 0.25},
    },
]


def _read_cluster_membership() -> pd.DataFrame:
    path = V09A_REPORT_ROOT / "cluster_membership.csv"
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path)
    if "month_start" in out.columns:
        out["month_start"] = pd.to_datetime(out["month_start"], utc=True, errors="coerce")
    return out


def _cluster_membership_asof(membership: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    if membership.empty or "month_start" not in membership.columns:
        return pd.DataFrame()
    month = pd.Timestamp(month_start)
    month = month.tz_localize("UTC") if month.tzinfo is None else month.tz_convert("UTC")
    months = pd.to_datetime(membership["month_start"], utc=True, errors="coerce")
    eligible_months = months[months <= month].dropna()
    if eligible_months.empty:
        return pd.DataFrame()
    use_month = eligible_months.max()
    return membership[months.eq(use_month)].copy()


def _add_cluster_context(prepared: pd.DataFrame) -> pd.DataFrame:
    if prepared.empty:
        return prepared.copy()
    if {"cluster_id", "cluster_impulse_density"}.issubset(prepared.columns):
        return prepared.copy()
    membership = _read_cluster_membership()
    if membership.empty:
        out = prepared.copy()
        out["cluster_id"] = ""
        out["cluster_size"] = np.nan
        out["cluster_impulse_density"] = np.nan
        return out
    out_frames: list[pd.DataFrame] = []
    data = prepared.copy()
    month_key = pd.to_datetime(
        pd.to_datetime(data["feature_time"], utc=True, errors="coerce").dt.strftime("%Y-%m-01"),
        utc=True,
        errors="coerce",
    )
    data["_cluster_month_start"] = month_key
    for month_start, group in data.groupby("_cluster_month_start", sort=False, dropna=False):
        local_membership = _cluster_membership_asof(membership, pd.Timestamp(month_start))
        local = group.drop(columns=["_cluster_month_start"], errors="ignore").copy()
        out_frames.append(add_cluster_graph_features(local, local_membership))
    return pd.concat(out_frames, ignore_index=True) if out_frames else data.drop(columns=["_cluster_month_start"], errors="ignore")


def add_v07d2_live_columns(df: pd.DataFrame, config: V07A2Config) -> pd.DataFrame:
    out = add_v07a2_live_columns(df, config).copy()
    out = _add_cluster_context(out)
    market = out.get("market_volume_impulse_density_high", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    density = pd.to_numeric(out.get("volume_impulse_density"), errors="coerce")
    beta_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    bucket = pd.Series("unknown", index=out.index, dtype="object")
    bucket[(beta_pct < 40)] = "beta_laggard"
    bucket[(beta_pct >= 40) & (beta_pct < 60)] = "beta_neutral"
    bucket[(beta_pct >= 60) & (beta_pct < 80)] = "beta_confirmed"
    bucket[(beta_pct >= 80) & (beta_pct < 95)] = "beta_extended"
    bucket[beta_pct >= 95] = "beta_extreme_overextended"
    out["c2_beta_extension_bucket"] = bucket
    out["c2_beta_extension_score"] = beta_pct
    out["c2_mir1_raw"] = market
    out["c2_bucket_beta_extreme_overextended"] = market & bucket.eq("beta_extreme_overextended")
    out["c2_relax_beta90"] = market & (beta_pct >= 90)
    out["c2_relax_density08_beta90"] = (density >= 0.08) & (beta_pct >= 90)
    out["c2_beta_continuation"] = market & bucket.isin(["beta_extended", "beta_extreme_overextended"])
    out["d2_mir1_intersect_cic1"] = out["c2_bucket_beta_extreme_overextended"]
    out["d2_mir1_intersect_cic2"] = out["c2_beta_continuation"]
    out["d2_mir1_only_ex_cic1"] = market & ~out["c2_bucket_beta_extreme_overextended"]
    out["d2_mir1_only_ex_cic2"] = market & ~out["c2_beta_continuation"]
    out["d2_mir1_missing_beta_state"] = market & bucket.eq("unknown")
    out["d2_mir1_or_cic1"] = market | out["c2_bucket_beta_extreme_overextended"]
    out["d2_mir1_or_cic2"] = market | out["c2_beta_continuation"]
    return out


def prepare_v07d2_features_from_history(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V07A2Config,
    days: int | None = 30,
) -> pd.DataFrame:
    prepared = prepare_v07a2_features_from_history(feature_path, instruments, base_config, config, days)
    return add_v07d2_live_columns(prepared, config)


def build_v07d2_paper_ledger(
    prepared: pd.DataFrame,
    config: V07A2Config,
    signal_start_time: pd.Timestamp | None = None,
    created_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = add_v07d2_live_columns(prepared, config)
    return build_v07a2_paper_ledger(prepared, config, signal_start_time, created_at)


def _candidate_trade_sample(trades: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    return trades[
        trades["candidate"].astype(str).eq(candidate)
        & trades["baseline_kind"].fillna("").astype(str).eq("")
    ].copy()


def _candidate_signal_sample(signals: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    return signals[signals["candidate"].astype(str).eq(candidate)].copy()


def _overlap_row(bucket: str, signals: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    exits = trades["exit_reason"].astype(str) if not trades.empty else pd.Series(dtype=str)
    holding = pd.to_numeric(trades.get("holding_minutes", pd.Series(dtype=float)), errors="coerce")
    return {
        "bucket": bucket,
        "signals": int(len(signals)),
        "filled_trades": int(len(trades)),
        "net10": _safe_float(trades.get("net_return_10bp", pd.Series(dtype=float)).mean()),
        "net20": _safe_float(trades.get("net_return_20bp", pd.Series(dtype=float)).mean()),
        "hit10_12h": np.nan,
        "hit10_during_trade": _safe_float((pd.to_numeric(trades.get("mfe", pd.Series(dtype=float)), errors="coerce") >= 0.10).mean()),
        "tp_rate": float(exits.str.startswith("tp").mean()) if not exits.empty else np.nan,
        "sl_rate": float(exits.str.startswith("sl").mean()) if not exits.empty else np.nan,
        "timeout_rate": float(exits.isin(["max_hold", "open"]).mean()) if not exits.empty else np.nan,
        "avg_holding_minutes": _safe_float(holding.mean()),
        "max_concurrent_positions": _safe_float(
            pd.to_numeric(trades.get("concurrent_positions_at_entry", pd.Series(dtype=float)), errors="coerce").max()
        ),
    }


def _shadow_base_signal_id(trades: pd.DataFrame) -> pd.Series:
    entry_time = pd.to_datetime(trades["local_volume_shock_time"], utc=True, errors="coerce").astype(str)
    return trades["exchange"].astype(str) + "|" + trades["symbol"].astype(str) + "|" + entry_time


def _shadow_trade_pool(trades: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    data = trades[
        trades["candidate"].astype(str).isin([str(item) for item in spec["candidates"]])
        & trades["baseline_kind"].fillna("").astype(str).eq("")
    ].copy()
    if data.empty:
        return data
    data["shadow_base_signal_id"] = _shadow_base_signal_id(data)
    data["candidate_priority"] = np.where(data["candidate"].astype(str).eq("CIC1_FILTERED_MIR1"), 2, 1)
    data["_cost_priority"] = 0
    data = data.sort_values(
        ["shadow_base_signal_id", "candidate_priority", "entry_time"],
        ascending=[True, False, True],
    )
    return data.drop_duplicates(["shadow_base_signal_id"], keep="first").copy()


def _live_burst_phase_bucket(count: int) -> str:
    if count <= 3:
        return "order_1_3"
    if count <= 8:
        return "order_4_8"
    if count <= 14:
        return "order_9_14"
    return "order_15_plus"


def _add_shadow_burst_phase(pool: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    out = pool.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out = out.sort_values(["entry_time", "symbol"]).copy()
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
    out["burst_count_so_far"] = out.groupby("burst_id", sort=False).cumcount() + 1
    starts = out.groupby("burst_id", sort=False)["entry_time"].transform("min")
    out["time_since_burst_start_so_far"] = (out["entry_time"] - starts).dt.total_seconds() / 60.0
    out["burst_phase_bucket"] = out["burst_count_so_far"].map(lambda value: _live_burst_phase_bucket(int(value)))
    out["asof_phase_passed"] = True
    out["uses_final_burst_size_for_decision"] = False
    return out


def _shadow_payload(row: Any, spec: dict[str, object], *, max_positions: int, concurrent_positions: int) -> dict[str, object]:
    payload = row._asdict()
    payload["portfolio_id"] = spec["portfolio_id"]
    payload["portfolio_pool"] = spec["pool"]
    payload["portfolio_description"] = spec["description"]
    payload["max_positions_at_time"] = max_positions
    payload["concurrent_positions"] = concurrent_positions
    payload["core_positions_at_time"] = concurrent_positions
    payload["overflow_positions_at_time"] = 0
    payload["is_core"] = True
    payload["is_overflow"] = False
    payload["sleeve"] = "core"
    payload["position_size"] = 1.0
    payload["extra_exposure"] = 0.0
    payload["overflow_reason"] = ""
    payload["overflow_slot_index"] = np.nan
    return payload


def _select_shadow_portfolio(pool: pd.DataFrame, spec: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pool.copy(), pool.copy()
    data = pool.copy()
    score_col = str(spec["score_col"])
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    data["exit_time"] = pd.to_datetime(data["exit_time"], utc=True, errors="coerce")
    if score_col == "__first_come__":
        data["rank_score"] = 0.0
    else:
        data["rank_score"] = pd.to_numeric(data.get(score_col), errors="coerce").fillna(-np.inf)
    data["rank_position"] = (
        data.groupby("entry_time", sort=False)["rank_score"].rank(ascending=False, method="first").astype(int)
    )
    data = data.sort_values(["entry_time", "rank_score", "candidate_priority", "symbol"], ascending=[True, False, False, True])
    active: list[tuple[pd.Timestamp, str]] = []
    selected_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    max_positions = int(spec["max_positions"])
    for row in data.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active = [(exit_time, symbol) for exit_time, symbol in active if exit_time > entry]
        active_symbols = {symbol for _, symbol in active}
        payload = _shadow_payload(row, spec, max_positions=max_positions, concurrent_positions=len(active))
        if str(row.symbol) in active_symbols:
            payload["selected"] = False
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if len(active) >= max_positions:
            payload["selected"] = False
            payload["skip_reason"] = "portfolio_full"
            skipped_rows.append(payload)
            continue
        payload["selected"] = True
        payload["skip_reason"] = ""
        selected_rows.append(payload)
        active.append((pd.Timestamp(row.exit_time), str(row.symbol)))
    return pd.DataFrame(selected_rows), pd.DataFrame(skipped_rows)


def _overflow_position_size(row: Any, spec: dict[str, object]) -> float:
    mapping = spec.get("overflow_size_by_candidate", {})
    if not isinstance(mapping, dict):
        return 0.0
    return float(mapping.get(str(getattr(row, "candidate", "")), 0.0))


def _select_overflow_shadow_portfolio(pool: pd.DataFrame, spec: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pool.copy(), pool.copy()
    data = _add_shadow_burst_phase(pool)
    data["rank_score"] = 0.0
    data["rank_position"] = data.groupby("entry_time", sort=False)["rank_score"].rank(ascending=False, method="first").astype(int)
    data = data.sort_values(["entry_time", "symbol", "candidate_priority"], ascending=[True, True, False])
    active_core: list[tuple[pd.Timestamp, str]] = []
    active_overflow: list[tuple[pd.Timestamp, str]] = []
    selected_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    core_max = int(spec.get("core_max_positions", spec.get("max_positions", 8)))
    overflow_max = int(spec.get("overflow_max_slots", 0))
    overflow_min_burst = int(spec.get("overflow_min_burst_count", 9))
    for row in data.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active_core = [(exit_time, symbol) for exit_time, symbol in active_core if exit_time > entry]
        active_overflow = [(exit_time, symbol) for exit_time, symbol in active_overflow if exit_time > entry]
        active_symbols = {symbol for _, symbol in [*active_core, *active_overflow]}
        payload = _shadow_payload(row, spec, max_positions=core_max, concurrent_positions=len(active_core) + len(active_overflow))
        payload["core_positions_at_time"] = len(active_core)
        payload["overflow_positions_at_time"] = len(active_overflow)
        if str(row.symbol) in active_symbols:
            payload["selected"] = False
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if len(active_core) < core_max:
            payload["selected"] = True
            payload["skip_reason"] = ""
            selected_rows.append(payload)
            active_core.append((pd.Timestamp(row.exit_time), str(row.symbol)))
            continue
        overflow_size = _overflow_position_size(row, spec)
        overflow_eligible = int(getattr(row, "burst_count_so_far", 0)) >= overflow_min_burst and overflow_size > 0
        if overflow_eligible and len(active_overflow) < overflow_max:
            payload["selected"] = True
            payload["skip_reason"] = ""
            payload["is_core"] = False
            payload["is_overflow"] = True
            payload["sleeve"] = "overflow"
            payload["position_size"] = overflow_size
            payload["extra_exposure"] = overflow_size
            payload["overflow_reason"] = "portfolio_full_late_burst_overflow"
            payload["overflow_slot_index"] = len(active_overflow) + 1
            selected_rows.append(payload)
            active_overflow.append((pd.Timestamp(row.exit_time), str(row.symbol)))
            continue
        payload["selected"] = False
        if overflow_eligible:
            payload["skip_reason"] = "overflow_full"
        else:
            payload["skip_reason"] = "portfolio_full_not_overflow_eligible"
        skipped_rows.append(payload)
    return pd.DataFrame(selected_rows), pd.DataFrame(skipped_rows)


def _net_col(cost: int) -> str:
    return f"net_return_{cost}bp"


def _portfolio_metric_row(
    portfolio_id: str,
    selected: pd.DataFrame,
    skipped: pd.DataFrame,
    *,
    cost: int,
) -> dict[str, object]:
    net = pd.to_numeric(selected.get(_net_col(cost), pd.Series(dtype=float)), errors="coerce")
    skipped_net = pd.to_numeric(skipped.get(_net_col(cost), pd.Series(dtype=float)), errors="coerce")
    exits = selected.get("exit_reason", pd.Series(dtype=str)).astype(str)
    return {
        "portfolio_id": portfolio_id,
        "cost_single_side_bps": cost,
        "selected_trades": int(len(selected)),
        "skipped_candidates": int(len(skipped)),
        "selected_net": _safe_float(net.mean()),
        "skipped_net": _safe_float(skipped_net.mean()),
        "selected_minus_skipped": _safe_float(net.mean() - skipped_net.mean()) if len(net) and len(skipped_net) else np.nan,
        "selected_hit_rate": _safe_float((net > 0).mean()) if len(net) else np.nan,
        "skipped_hit_rate": _safe_float((skipped_net > 0).mean()) if len(skipped_net) else np.nan,
        "tp_rate": _safe_float(exits.str.startswith("tp").mean()) if len(exits) else np.nan,
        "sl_rate": _safe_float(exits.str.startswith("sl").mean()) if len(exits) else np.nan,
        "timeout_rate": _safe_float(exits.isin(["max_hold", "open"]).mean()) if len(exits) else np.nan,
        "selected_avg_rank_score": _safe_float(pd.to_numeric(selected.get("rank_score", pd.Series(dtype=float)), errors="coerce").mean()),
        "skipped_avg_rank_score": _safe_float(pd.to_numeric(skipped.get("rank_score", pd.Series(dtype=float)), errors="coerce").mean()),
    }


def shadow_portfolio_live(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    skipped_frames = []
    summary_rows = []
    daily_rows = []
    status_rows = []
    for spec in SHADOW_PORTFOLIOS:
        pool = _shadow_trade_pool(trades, spec)
        if spec.get("selection") == "late_burst_overflow":
            selected, skipped = _select_overflow_shadow_portfolio(pool, spec)
        else:
            selected, skipped = _select_shadow_portfolio(pool, spec)
        for frame, selected_flag in [(selected, True), (skipped, False)]:
            if frame.empty:
                continue
            frame = frame.copy()
            frame["selected"] = selected_flag
            frame["net10_if_taken"] = pd.to_numeric(frame.get("net_return_10bp"), errors="coerce")
            frame["net20_if_taken"] = pd.to_numeric(frame.get("net_return_20bp"), errors="coerce")
            if selected_flag:
                selected_frames.append(frame)
            else:
                skipped_frames.append(frame)
        for cost in [10, 20, 30, 50]:
            summary_rows.append(_portfolio_metric_row(str(spec["portfolio_id"]), selected, skipped, cost=cost))
        date_sources = [
            frame["entry_time"]
            for frame in [selected, skipped]
            if not frame.empty and "entry_time" in frame.columns
        ]
        date_values = (
            pd.to_datetime(pd.concat(date_sources), utc=True, errors="coerce").dt.date.dropna().unique()
            if date_sources
            else []
        )
        for day in sorted(date_values):
            sel_day = selected[pd.to_datetime(selected.get("entry_time"), utc=True, errors="coerce").dt.date.eq(day)] if not selected.empty else selected
            skip_day = skipped[pd.to_datetime(skipped.get("entry_time"), utc=True, errors="coerce").dt.date.eq(day)] if not skipped.empty else skipped
            row = _portfolio_metric_row(str(spec["portfolio_id"]), sel_day, skip_day, cost=20)
            row["date"] = day
            daily_rows.append(row)
        status_rows.append(
            {
                "portfolio_id": spec["portfolio_id"],
                "pool": spec["pool"],
                "max_positions": spec["max_positions"],
                "ranking": spec.get("ranking", "cluster_impulse_density_high"),
                "candidate_count": int(len(pool)),
                "selected_trades": int(len(selected)),
                "skipped_candidates": int(len(skipped)),
                "core_trades": int(selected.get("is_core", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
                if not selected.empty
                else 0,
                "overflow_trades": int(selected.get("is_overflow", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
                if not selected.empty
                else 0,
            }
        )
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    daily = pd.DataFrame(daily_rows)
    status = pd.DataFrame(status_rows)
    return status, selected_all, skipped_all, daily, summary


def _overflow_trade_ledger(shadow_trades: pd.DataFrame) -> pd.DataFrame:
    if shadow_trades.empty or "portfolio_id" not in shadow_trades.columns:
        return pd.DataFrame()
    ledger = shadow_trades[shadow_trades["portfolio_id"].astype(str).eq(OVERFLOW_PORTFOLIO_ID)].copy()
    if ledger.empty:
        return ledger
    ledger["entry_time"] = pd.to_datetime(ledger["entry_time"], utc=True, errors="coerce")
    ledger["exit_time"] = pd.to_datetime(ledger["exit_time"], utc=True, errors="coerce")
    ledger["position_size"] = pd.to_numeric(ledger.get("position_size", 1.0), errors="coerce").fillna(1.0)
    ledger["is_core"] = ledger.get("is_core", False).fillna(False).astype(bool)
    ledger["is_overflow"] = ledger.get("is_overflow", False).fillna(False).astype(bool)
    ledger["extra_exposure"] = np.where(ledger["is_overflow"], ledger["position_size"], 0.0)
    if "trade_id" not in ledger.columns:
        ledger["trade_id"] = (
            ledger["portfolio_id"].astype(str)
            + "|"
            + ledger["symbol"].astype(str)
            + "|"
            + ledger["entry_time"].astype(str)
        )
    preferred = [
        "trade_id",
        "signal_id",
        "symbol",
        "candidate",
        "candidate_type",
        "is_core",
        "is_overflow",
        "sleeve",
        "overflow_reason",
        "burst_id",
        "burst_count_so_far",
        "burst_phase_bucket",
        "overflow_slot_index",
        "position_size",
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_return_10bp",
        "net_return_20bp",
        "net_return_30bp",
        "extra_exposure",
        "core_positions_at_time",
        "overflow_positions_at_time",
        "concurrent_positions",
    ]
    cols = [col for col in preferred if col in ledger.columns]
    rest = [col for col in ledger.columns if col not in cols]
    return ledger[cols + rest].sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _weighted_return_sum(sample: pd.DataFrame, cost: int) -> float:
    if sample.empty:
        return 0.0
    net = pd.to_numeric(sample.get(_net_col(cost), pd.Series(dtype=float)), errors="coerce")
    weight = pd.to_numeric(sample.get("position_size", pd.Series(1.0, index=sample.index)), errors="coerce").fillna(1.0)
    return float((net * weight).sum())


def _weighted_avg_return(sample: pd.DataFrame, cost: int) -> float:
    if sample.empty:
        return np.nan
    weight = pd.to_numeric(sample.get("position_size", pd.Series(1.0, index=sample.index)), errors="coerce").fillna(1.0)
    denom = float(weight.sum())
    return _weighted_return_sum(sample, cost) / denom if denom else np.nan


def _overflow_summary_row(
    ledger: pd.DataFrame,
    *,
    date: object | None = None,
    cost: int = 20,
    core_denominator: int = 8,
) -> dict[str, object]:
    core = ledger[ledger.get("is_core", pd.Series(dtype=bool)).fillna(False).astype(bool)] if not ledger.empty else ledger
    overflow = ledger[ledger.get("is_overflow", pd.Series(dtype=bool)).fillna(False).astype(bool)] if not ledger.empty else ledger
    core_net = pd.to_numeric(core.get(_net_col(cost), pd.Series(dtype=float)), errors="coerce")
    overflow_exposure = float(pd.to_numeric(overflow.get("position_size", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    overflow_weighted = _weighted_return_sum(overflow, cost)
    combined_weighted = _weighted_return_sum(ledger, cost)
    row = {
        "date": date if date is not None else "ALL",
        "portfolio_id": OVERFLOW_PORTFOLIO_ID,
        "cost_single_side_bps": cost,
        "core_trades": int(len(core)),
        "overflow_trades": int(len(overflow)),
        "combined_trades": int(len(ledger)),
        "core_net": _safe_float(core_net.mean()),
        "overflow_net": _weighted_avg_return(overflow, cost),
        "combined_net_per_core_capacity": combined_weighted / max(1, core_denominator),
        "extra_exposure": overflow_exposure,
        "overflow_weighted_return_sum": overflow_weighted,
        "incremental_return_per_extra_exposure": overflow_weighted / overflow_exposure if overflow_exposure else np.nan,
        "real_live_allowed": False,
    }
    return row


def _overflow_daily_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if ledger.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "portfolio_id",
                "cost_single_side_bps",
                "core_trades",
                "overflow_trades",
                "combined_trades",
                "core_net",
                "overflow_net",
                "combined_net_per_core_capacity",
                "extra_exposure",
                "overflow_weighted_return_sum",
                "incremental_return_per_extra_exposure",
                "real_live_allowed",
            ]
        )
    data = ledger.copy()
    data["date"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce").dt.date
    for day, group in data.groupby("date", sort=True, dropna=False):
        for cost in [10, 20, 30]:
            rows.append(_overflow_summary_row(group, date=day, cost=cost))
    for cost in [10, 20, 30]:
        rows.append(_overflow_summary_row(data, date="ALL", cost=cost))
    return pd.DataFrame(rows)


def _write_overflow_status(report_root: Path, ledger: pd.DataFrame, daily: pd.DataFrame) -> None:
    all20 = daily[
        daily["date"].astype(str).eq("ALL") & pd.to_numeric(daily["cost_single_side_bps"], errors="coerce").eq(20)
    ] if not daily.empty else pd.DataFrame()
    row = all20.iloc[0].to_dict() if not all20.empty else {}
    overflow_trades = int(row.get("overflow_trades", 0) or 0)
    if overflow_trades < 20:
        sample_status = "system_check_only"
        evaluation_status = "no_decision"
    elif overflow_trades < 50:
        sample_status = "behavior_check"
        evaluation_status = "no_upgrade_no_downgrade"
    else:
        sample_status = "candidate_check"
        evaluation_status = "evaluate_incremental_overflow"
    lines = [
        "# v1.0D.2 Late-Burst Overflow Shadow",
        "",
        "- portfolio_id: P2_MAX8_PLUS_O6_LATE_BURST_OVERFLOW",
        "- status: shadow_only",
        "- real_live_allowed: false",
        "- core: P2 CIC1+CIC2 combined max8, size=1.0",
        "- overflow_trigger: portfolio_full and burst_count_so_far >= 9",
        "- overflow_slots: 4",
        "- overflow_size: CIC1=0.50, CIC2=0.25",
        "- exit: vol_regime_fast",
        "",
        "## 20bp Summary",
        f"- core_trades: {int(row.get('core_trades', 0) or 0)}",
        f"- overflow_trades: {overflow_trades}",
        f"- combined_trades: {int(row.get('combined_trades', 0) or 0)}",
        f"- core_net20: {row.get('core_net', np.nan):.4%}" if pd.notna(row.get("core_net", np.nan)) else "- core_net20: n/a",
        f"- overflow_net20: {row.get('overflow_net', np.nan):.4%}" if pd.notna(row.get("overflow_net", np.nan)) else "- overflow_net20: n/a",
        f"- combined_net20_per_core_capacity: {row.get('combined_net_per_core_capacity', np.nan):.4%}"
        if pd.notna(row.get("combined_net_per_core_capacity", np.nan))
        else "- combined_net20_per_core_capacity: n/a",
        f"- extra_exposure: {row.get('extra_exposure', 0.0):.4f}",
        f"- incremental_return_per_extra_exposure: {row.get('incremental_return_per_extra_exposure', np.nan):.4%}"
        if pd.notna(row.get("incremental_return_per_extra_exposure", np.nan))
        else "- incremental_return_per_extra_exposure: n/a",
        f"- sample_status: {sample_status}",
        f"- evaluation_status: {evaluation_status}",
    ]
    if not ledger.empty:
        last_entry = pd.to_datetime(ledger["entry_time"], utc=True, errors="coerce").max()
        lines.extend(["", f"- latest_entry_time: {last_entry}"])
    (report_root / "overflow_current_status.md").write_text("\n".join(lines), encoding="utf-8")


def mir1_cic_overlap_live(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    partition: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mapping = [
        ("MIR1_raw", "MIR1_RAW"),
        ("CIC1", "CIC1_FILTERED_MIR1"),
        ("CIC2", "CIC2_FILTERED_MIR1"),
        ("MIR1_intersect_CIC1", "CIC1_FILTERED_MIR1"),
        ("MIR1_only_ex_CIC1", "MIR1_ONLY_EX_CIC1"),
        ("MIR1_intersect_CIC2", "CIC2_FILTERED_MIR1"),
        ("MIR1_only_ex_CIC2", "MIR1_ONLY_EX_CIC2"),
    ]
    rows = []
    for bucket, candidate in mapping:
        rows.append(_overlap_row(bucket, _candidate_signal_sample(signals, candidate), _candidate_trade_sample(trades, candidate)))
    out = pd.DataFrame(rows)
    if partition is not None and not partition.empty:
        event_map = {
            "MIR1_raw": "MIR1_raw_events",
            "CIC1": "MIR1_intersect_CIC1_events",
            "CIC2": "MIR1_intersect_CIC2_events",
            "MIR1_intersect_CIC1": "MIR1_intersect_CIC1_events",
            "MIR1_only_ex_CIC1": "MIR1_only_ex_CIC1_events",
            "MIR1_intersect_CIC2": "MIR1_intersect_CIC2_events",
            "MIR1_only_ex_CIC2": "MIR1_only_ex_CIC2_events",
        }
        counts = partition.set_index("partition")["feature_events"].to_dict()
        out["signals"] = out["bucket"].map(lambda bucket: int(counts.get(event_map.get(str(bucket), ""), 0)))
    return out


def overlap_partition_audit(prepared: pd.DataFrame, signals: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if prepared.empty:
        return pd.DataFrame()
    data = prepared.copy()
    top_rank = pd.to_numeric(data.get("dynamic_all_rank"), errors="coerce")
    event = data.get("bullish_volume_shock_event", pd.Series(False, index=data.index)).fillna(False).astype(bool)
    mir1 = data.get("c2_mir1_raw", pd.Series(False, index=data.index)).fillna(False).astype(bool) & event & (top_rank <= 50)
    cic1 = data.get("c2_bucket_beta_extreme_overextended", pd.Series(False, index=data.index)).fillna(False).astype(bool) & event & (top_rank <= 50)
    cic2 = data.get("c2_beta_continuation", pd.Series(False, index=data.index)).fillna(False).astype(bool) & event & (top_rank <= 50)
    bucket = data.get("c2_beta_extension_bucket", pd.Series("unknown", index=data.index)).astype(str)
    partitions = {
        "MIR1_raw_events": mir1,
        "MIR1_intersect_CIC1_events": mir1 & cic1,
        "MIR1_only_ex_CIC1_events": mir1 & ~cic1 & ~bucket.eq("unknown"),
        "MIR1_missing_beta_state_CIC1_events": mir1 & bucket.eq("unknown"),
        "MIR1_intersect_CIC2_events": mir1 & cic2,
        "MIR1_only_ex_CIC2_events": mir1 & ~cic2 & ~bucket.eq("unknown"),
        "MIR1_missing_beta_state_CIC2_events": mir1 & bucket.eq("unknown"),
    }
    rows = []
    for name, mask in partitions.items():
        rows.append({"partition": name, "feature_events": int(mask.sum())})
    audit = pd.DataFrame(rows)
    signal_counts = signals.groupby("candidate", dropna=False).size().to_dict() if not signals.empty else {}
    trade_counts = trades[trades["baseline_kind"].fillna("").eq("")].groupby("candidate", dropna=False).size().to_dict() if not trades.empty else {}
    signal_map = {
        "MIR1_raw_events": "MIR1_RAW",
        "MIR1_intersect_CIC1_events": "CIC1_FILTERED_MIR1",
        "MIR1_only_ex_CIC1_events": "MIR1_ONLY_EX_CIC1",
        "MIR1_intersect_CIC2_events": "CIC2_FILTERED_MIR1",
        "MIR1_only_ex_CIC2_events": "MIR1_ONLY_EX_CIC2",
    }
    audit["candidate"] = audit["partition"].map(signal_map).fillna("")
    audit["paper_signals"] = audit["candidate"].map(signal_counts).fillna(0).astype(int)
    audit["paper_trades"] = audit["candidate"].map(trade_counts).fillna(0).astype(int)
    return audit


def _write_status(
    report_root: Path,
    prepared: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    config: V07A2Config,
    shadow_status: pd.DataFrame | None = None,
    shadow_summary: pd.DataFrame | None = None,
) -> None:
    latest = _latest_feature_time(prepared)
    market = _latest_market_state(prepared)
    primary = _primary_portfolio_trades(trades, config)
    net10 = _safe_float(primary.get("net_return_10bp", pd.Series(dtype=float)).mean())
    net20 = _safe_float(primary.get("net_return_20bp", pd.Series(dtype=float)).mean())
    sample_status, evaluation_status = _sample_status(len(primary))
    current_lines = [
        "# v0.7D.2 CIC-filtered MIR1 Paper-Live Status",
        "",
        f"- strategy_id: {config.experiment.strategy_id}",
        f"- primary_candidate: {config.experiment.primary_candidate}",
        f"- candidate_rating: {config.experiment.candidate_rating}",
        f"- latest_feature_time: {latest}",
        f"- btc_state: {market['btc_state']}",
        f"- volume_impulse_density: {market['volume_impulse_density']}",
        f"- market_volume_impulse_density_high: {market['market_volume_impulse_density_high']}",
        f"- data_stale: {_data_stale(prepared, config)}",
        f"- paper_live_allowed: {config.experiment.paper_live_allowed}",
        f"- real_live_allowed: {config.experiment.real_live_allowed}",
        f"- tick_validation_status: {config.experiment.tick_validation_status}",
        f"- historical_validation_status: {config.experiment.historical_validation_status}",
        f"- total_candidate_signals: {len(signals)}",
        f"- primary_portfolio_trades: {len(primary)}",
        f"- sample_status: {sample_status}",
        f"- evaluation_status: {evaluation_status}",
        f"- primary_net_10bp_avg: {net10:.4%}" if pd.notna(net10) else "- primary_net_10bp_avg: n/a",
        f"- primary_net_20bp_avg: {net20:.4%}" if pd.notna(net20) else "- primary_net_20bp_avg: n/a",
        f"- baseline_trades: {len(baseline_trades)}",
    ]
    if shadow_status is not None and not shadow_status.empty:
        current_lines.extend(["", "## Shadow Portfolios"])
        for row in shadow_status.itertuples(index=False):
            current_lines.append(
                f"- {row.portfolio_id}: selected={row.selected_trades}, skipped={row.skipped_candidates}, "
                f"pool={row.pool}, max_positions={row.max_positions}"
            )
    (report_root / "current_status.md").write_text("\n".join(current_lines), encoding="utf-8")
    summary = _summary(signals, trades, ["candidate", "candidate_role"])
    primary_summary = _summary(signals, trades, ["candidate", "candidate_role"], accepted_only=True)
    baseline_summary = _summary(pd.DataFrame(), baseline_trades, ["candidate", "baseline_kind"])
    note = [
        "# v0.7D.2 Candidate Status",
        "",
        "Decision: CIC-filtered MIR1 is the primary B- paper-live candidate.",
        "",
        "Definition: market volume impulse density high -> local bullish volume shock -> beta continuation state -> 1pct reclaim -> vol-regime-fast exit.",
        "",
        "Primary: CIC1_FILTERED_MIR1. Secondary shadow: CIC2_FILTERED_MIR1. MIR1_RAW is reference/ablation only.",
        "",
        "Gate rule: signal and entry must both pass using as-of market/continuation features (`feature_time <= decision_time`).",
        "",
        "Real-live: disabled until tick/orderflow validation and sufficient future paper-live sample.",
        "",
        "Do not evaluate before primary filled trades >= 30 for behavior checks and >= 100 for candidate checks.",
        "",
        f"filled_trades = {len(primary)}",
        f"sample_status = {sample_status}",
        f"evaluation_status = {evaluation_status}",
        "",
        "## Primary Portfolio",
    ]
    if primary_summary.empty:
        note.append("- No primary portfolio trades yet.")
    else:
        for row in primary_summary.itertuples(index=False):
            note.append(
                f"- {row.candidate}: trades={row.trades}, net10={getattr(row, 'net_10bp_avg', np.nan):.4%}, "
                f"net20={getattr(row, 'net_20bp_avg', np.nan):.4%}"
            )
    note.append("")
    note.append("## Candidate / Ablation Rows")
    if summary.empty:
        note.append("- No candidate rows yet.")
    else:
        for row in summary.itertuples(index=False):
            note.append(f"- {row.candidate}/{row.candidate_role}: trades={row.trades}, net10={getattr(row, 'net_10bp_avg', np.nan):.4%}")
    note.append("")
    note.append("## Baseline Shadows")
    if baseline_summary.empty:
        note.append("- No baseline shadow trades yet.")
    else:
        for row in baseline_summary.itertuples(index=False):
            note.append(f"- {row.candidate}/{row.baseline_kind}: trades={row.trades}, net10={getattr(row, 'net_10bp_avg', np.nan):.4%}")
    note.append("")
    note.append("## v0.9B.1 Shadow Portfolio Integration")
    note.append("Primary remains unchanged. Shadow portfolios are counterfactual ledgers and do not affect primary paper PnL.")
    if shadow_summary is None or shadow_summary.empty:
        note.append("- No shadow portfolio candidates yet.")
    else:
        focal = shadow_summary[pd.to_numeric(shadow_summary["cost_single_side_bps"], errors="coerce").eq(20)].copy()
        for row in focal.itertuples(index=False):
            note.append(
                f"- {row.portfolio_id}: selected={row.selected_trades}, skipped={row.skipped_candidates}, "
                f"selected_net20={row.selected_net:.4%}, skipped_net20={row.skipped_net:.4%}, "
                f"delta={row.selected_minus_skipped:.4%}"
            )
    (report_root / "candidate_status.md").write_text("\n".join(note), encoding="utf-8")


def _write_shadow_status(
    report_root: Path,
    shadow_status: pd.DataFrame,
    shadow_summary: pd.DataFrame,
) -> None:
    lines = [
        "# v0.9B.1 Shadow Portfolio Status",
        "",
        "These portfolios are diagnostics only. They do not replace CIC-filtered MIR1 primary.",
        "",
    ]
    if shadow_status.empty:
        lines.append("- No shadow portfolio candidates yet.")
    else:
        for row in shadow_status.itertuples(index=False):
            lines.append(
                f"- {row.portfolio_id}: pool={row.pool}, ranking={row.ranking}, max_positions={row.max_positions}, "
                f"selected={row.selected_trades}, skipped={row.skipped_candidates}"
            )
    if not shadow_summary.empty:
        lines.extend(["", "## 20bp Counterfactual"])
        focal = shadow_summary[pd.to_numeric(shadow_summary["cost_single_side_bps"], errors="coerce").eq(20)].copy()
        for row in focal.itertuples(index=False):
            lines.append(
                f"- {row.portfolio_id}: selected_net20={row.selected_net:.4%}, "
                f"skipped_net20={row.skipped_net:.4%}, selected_minus_skipped={row.selected_minus_skipped:.4%}"
            )
    (report_root / "portfolio_current_status.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07d2_outputs(
    prepared: pd.DataFrame,
    config: V07A2Config,
    signal_days: int,
    report_root: Path = REPORT_ROOT,
    paper_data_root: Path = PAPER_DATA_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    paper_data_root = ensure_dir(paper_data_root)
    prepared = add_v07d2_live_columns(prepared, config)
    latest = _latest_feature_time(prepared)
    signal_start_time = latest - pd.Timedelta(days=signal_days) if pd.notna(latest) else None
    signals, trades, baseline_signals, baseline_trades = build_v07d2_paper_ledger(prepared, config, signal_start_time)
    daily = _daily_summary(signals, trades, config)
    candidate_summary = _summary(signals, trades, ["candidate", "candidate_role"])
    primary_summary = _summary(signals, trades, ["candidate", "candidate_role"], accepted_only=True)
    baseline_summary = _summary(baseline_signals, baseline_trades, ["candidate", "baseline_kind"])
    skipped = signals[signals["status"].eq("skipped")].copy() if not signals.empty else signals.copy()
    partition = overlap_partition_audit(prepared, signals, trades)
    overlap = mir1_cic_overlap_live(signals, trades, partition)
    shadow_status, shadow_trades, shadow_skipped, shadow_daily, shadow_summary = shadow_portfolio_live(trades)
    overflow_ledger = _overflow_trade_ledger(shadow_trades)
    overflow_daily = _overflow_daily_summary(overflow_ledger)
    outputs = {
        "paper_signals": report_root / "paper_signals.parquet",
        "paper_trades": report_root / "paper_trades.parquet",
        "baseline_signals": report_root / "baseline_signals.parquet",
        "baseline_trades": report_root / "baseline_trades.parquet",
        "daily_summary": report_root / "daily_summary.csv",
        "candidate_summary": report_root / "candidate_summary.csv",
        "primary_summary": report_root / "primary_summary.csv",
        "baseline_summary": report_root / "baseline_summary.csv",
        "skipped_signals": report_root / "skipped_signals.csv",
        "market_gate_audit": report_root / "market_gate_audit.csv",
        "shadow_baselines_live": report_root / "shadow_baselines_live.csv",
        "mir1_cic_overlap_live": report_root / "mir1_cic_overlap_live.csv",
        "overlap_partition_audit": report_root / "overlap_partition_audit.csv",
        "paper_portfolios": report_root / "paper_portfolios.parquet",
        "paper_portfolio_trades": report_root / "paper_portfolio_trades.parquet",
        "paper_skipped_candidates": report_root / "paper_skipped_candidates.parquet",
        "portfolio_daily_summary": report_root / "portfolio_daily_summary.csv",
        "portfolio_current_status": report_root / "portfolio_current_status.md",
        "ranking_shadow_summary": report_root / "ranking_shadow_summary.csv",
        "overflow_trade_ledger": report_root / "overflow_trade_ledger.parquet",
        "overflow_daily_summary": report_root / "overflow_daily_summary.csv",
        "overflow_current_status": report_root / "overflow_current_status.md",
        "current_status": report_root / "current_status.md",
        "candidate_status": report_root / "candidate_status.md",
        "decision_log": report_root / "decision_log.md",
        "paper_signals_data": paper_data_root / "paper_signals.parquet",
        "paper_trades_data": paper_data_root / "paper_trades.parquet",
    }
    write_parquet(signals, outputs["paper_signals"])
    write_parquet(trades, outputs["paper_trades"])
    write_parquet(baseline_signals, outputs["baseline_signals"])
    write_parquet(baseline_trades, outputs["baseline_trades"])
    write_parquet(signals, outputs["paper_signals_data"])
    write_parquet(trades, outputs["paper_trades_data"])
    daily.to_csv(outputs["daily_summary"], index=False)
    candidate_summary.to_csv(outputs["candidate_summary"], index=False)
    primary_summary.to_csv(outputs["primary_summary"], index=False)
    baseline_summary.to_csv(outputs["baseline_summary"], index=False)
    skipped.to_csv(outputs["skipped_signals"], index=False)
    gate_audit = _market_gate_audit(signals, trades, prepared)
    if not gate_audit.empty and "candidate" in gate_audit.columns:
        gate_audit["is_primary"] = gate_audit["candidate"].astype(str).eq(config.experiment.primary_candidate)
    gate_audit.to_csv(outputs["market_gate_audit"], index=False)
    _shadow_baselines_live(signals, trades, baseline_signals, baseline_trades, config).to_csv(outputs["shadow_baselines_live"], index=False)
    overlap.to_csv(outputs["mir1_cic_overlap_live"], index=False)
    partition.to_csv(outputs["overlap_partition_audit"], index=False)
    write_parquet(shadow_status, outputs["paper_portfolios"])
    write_parquet(shadow_trades, outputs["paper_portfolio_trades"])
    write_parquet(shadow_skipped, outputs["paper_skipped_candidates"])
    shadow_daily.to_csv(outputs["portfolio_daily_summary"], index=False)
    shadow_summary.to_csv(outputs["ranking_shadow_summary"], index=False)
    write_parquet(overflow_ledger, outputs["overflow_trade_ledger"])
    overflow_daily.to_csv(outputs["overflow_daily_summary"], index=False)
    _write_shadow_status(report_root, shadow_status, shadow_summary)
    _write_overflow_status(report_root, overflow_ledger, overflow_daily)
    _write_status(report_root, prepared, signals, trades, baseline_trades, config, shadow_status, shadow_summary)
    _append_decision_log(report_root, prepared, trades, config)
    return outputs


def write_v07d2_cic_mir1_paper_live(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V07A2Config,
    report_root: Path = REPORT_ROOT,
    paper_data_root: Path = PAPER_DATA_ROOT,
    days: int | None = 30,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    paper_data_root = ensure_dir(paper_data_root)
    prepared = prepare_v07d2_features_from_history(feature_path, instruments, base_config, config, days)
    signal_days = days if days is not None else 3650
    return write_v07d2_outputs(prepared, config, signal_days, report_root, paper_data_root)


S2_PAPER_LIVE_CANDIDATES = (
    "S2_FAIL_CIC2_BREAK_LOW",
    "S2_FAIL_CIC2_NO_RECLAIM_BREAK_LOW",
)


S2_PAPER_LIVE_ROOT = Path("reports/v1_0_short_paper_live")


def write_v07d2_s2_paper_live(
    prepared: pd.DataFrame,
    signal_days: int = 7,
    report_root: Path | str | None = None,
) -> dict[str, Path]:
    """Roll v10 S2 long-failure-to-short candidates as a paper-live shadow.

    P0b contract:
    - Reuses ``simulate_short_candidate`` with the asymmetric short rule and
      the funding blocker (rejects entries when funding is too negative).
    - Filters trades to the latest ``signal_days`` window for status reporting.
    - Writes ``s2_candidate_summary.csv``, ``s2_paper_trades.parquet`` and an
      ``s2_current_status.md`` summary so the existing v07d2 primary status
      file is left untouched.
    """
    from pressure_graph.reports.v10_short_mirror import (
        CANDIDATES,
        _funding_block_short,
        _vol_regime_rule_short,
        add_short_mirror_columns,
        simulate_short_candidate,
    )

    if report_root is None:
        root = S2_PAPER_LIVE_ROOT
    else:
        root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)

    data = add_short_mirror_columns(prepared)
    latest = pd.to_datetime(data.get("feature_time"), utc=True, errors="coerce").max()
    signal_start = (
        latest - pd.Timedelta(days=signal_days) if pd.notna(latest) else None
    )

    s2_candidates = [c for c in CANDIDATES if c.candidate in S2_PAPER_LIVE_CANDIDATES]
    frames: list[pd.DataFrame] = []
    for candidate in s2_candidates:
        frames.append(
            simulate_short_candidate(
                data,
                candidate,
                rule_factory=_vol_regime_rule_short,
                funding_blocker=_funding_block_short,
            )
        )
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not trades.empty and signal_start is not None:
        trades = trades[
            pd.to_datetime(trades["signal_time"], utc=True, errors="coerce") >= signal_start
        ].copy()

    summary_rows: list[dict[str, object]] = []
    for candidate in s2_candidates:
        cand_trades = (
            trades[trades["candidate"].eq(candidate.candidate)]
            if not trades.empty
            else trades
        )
        n_trades = int(len(cand_trades))
        net20_series = pd.to_numeric(cand_trades.get("net20", pd.Series(dtype=float)), errors="coerce")
        sample_status, evaluation_status = _sample_status(n_trades)
        summary_rows.append(
            {
                "candidate": candidate.candidate,
                "family": candidate.family,
                "trades": n_trades,
                "net20": _safe_float(net20_series.mean()),
                "tp_rate": _safe_float(
                    (cand_trades.get("exit_reason", pd.Series(dtype=str)).astype(str).str.startswith("tp")).mean()
                )
                if n_trades
                else np.nan,
                "sample_status": sample_status,
                "evaluation_status": evaluation_status,
            }
        )
    summary = pd.DataFrame(summary_rows)

    outputs = {
        "candidate_summary": root / "s2_candidate_summary.csv",
        "paper_trades": root / "s2_paper_trades.parquet",
        "current_status": root / "s2_current_status.md",
    }
    summary.to_csv(outputs["candidate_summary"], index=False)
    if trades.empty:
        pd.DataFrame().to_parquet(outputs["paper_trades"], index=False)
    else:
        trades.to_parquet(outputs["paper_trades"], index=False)

    lines = [
        "# v1.0 S2 long-failure paper-live shadow",
        "",
        f"- latest_feature_time: {latest}",
        f"- signal_days: {signal_days}",
        f"- candidates: {len(s2_candidates)}",
        f"- total_trades: {int(len(trades))}",
        "",
        "## Candidates",
    ]
    for row in summary.itertuples(index=False):
        net20_text = "n/a" if pd.isna(row.net20) else f"{float(row.net20):.4%}"
        lines.append(
            f"- {row.candidate}: trades={int(row.trades)}, net20={net20_text}, "
            f"sample_status={row.sample_status}"
        )
    outputs["current_status"].write_text("\n".join(lines), encoding="utf-8")
    return outputs


__all__ = [
    "OVERFLOW_PORTFOLIO_ID",
    "PAPER_DATA_ROOT",
    "REPORT_ROOT",
    "S2_PAPER_LIVE_CANDIDATES",
    "S2_PAPER_LIVE_ROOT",
    "add_v07d2_live_columns",
    "build_v07d2_paper_ledger",
    "mir1_cic_overlap_live",
    "overlap_partition_audit",
    "prepare_v07d2_features_from_history",
    "write_v07d2_cic_mir1_paper_live",
    "write_v07d2_outputs",
    "write_v07d2_s2_paper_live",
]
