from __future__ import annotations

from pathlib import Path

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
        payload = row._asdict()
        payload["portfolio_id"] = spec["portfolio_id"]
        payload["portfolio_pool"] = spec["pool"]
        payload["portfolio_description"] = spec["description"]
        payload["max_positions_at_time"] = max_positions
        payload["concurrent_positions"] = len(active)
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
            }
        )
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    daily = pd.DataFrame(daily_rows)
    status = pd.DataFrame(status_rows)
    return status, selected_all, skipped_all, daily, summary


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
    _write_shadow_status(report_root, shadow_status, shadow_summary)
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


__all__ = [
    "PAPER_DATA_ROOT",
    "REPORT_ROOT",
    "add_v07d2_live_columns",
    "build_v07d2_paper_ledger",
    "mir1_cic_overlap_live",
    "overlap_partition_audit",
    "prepare_v07d2_features_from_history",
    "write_v07d2_cic_mir1_paper_live",
    "write_v07d2_outputs",
]
