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
from pressure_graph.reports.v22b_preentry_meta_router import (
    V21G_ROOT as V22_ROUTER_TRAIN_ROOT,
    V22BConfig,
    _feature_columns as _router_feature_columns,
    _fit_logistic as _router_fit_logistic,
    _predict_logistic as _router_predict_logistic,
    _read_csv as _router_read_csv,
    _train_ready as _router_train_ready,
)
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
CHECKPOINT_PORTFOLIO_IDS = {
    "S0": "P2_MAX8_BASELINE",
    "S1": "P2_MAX8_PLUS_O6",
    "S2": "P2_MAX8_CP60",
    "S3": "P2_MAX8_CP60_PLUS_O6",
    "S4": "P2_MAX8_CP60_PROTECT_A_CAP2",
    "S5": "P2_MAX8_CP60_PROTECT_A_CAP2_PLUS_O6",
}
CHECKPOINT_MINUTES = 60
CHECKPOINT_TRIGGER_COST_BPS = 20
CP60_PROTECT_A_BETA_HIGH_THRESHOLD = 99.97866821289062
CP60_PROTECT_A_CAP_PER_BURST = 2
PRE_ENTRY_ROUTER_THRESHOLDS = (0.70, 0.75, 0.80)
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


def _p2_combined_spec() -> dict[str, object]:
    for spec in SHADOW_PORTFOLIOS:
        if spec.get("portfolio_id") == "P2_CIC_COMBINED_BASKET_MAX8":
            return spec
    return {
        "portfolio_id": "P2_CIC_COMBINED_BASKET_MAX8",
        "pool": "CIC1_CIC2_COMBINED",
        "description": "fallback P2 combined basket spec",
        "candidates": ["CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"],
        "max_positions": 8,
        "score_col": "__first_come__",
    }


def _router_live_feature_frame(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()
    data = _add_shadow_burst_phase(pool).copy()
    entry = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    candidate = data["candidate"].astype(str)
    cic_type = np.where(candidate.eq("CIC1_FILTERED_MIR1"), "CIC1", "CIC2")
    out = pd.DataFrame(
        {
            "trade_key": data.get("signal_id", data.index.astype(str)).astype(str),
            "signal_id": data.get("signal_id", data.index.astype(str)).astype(str),
            "trade_id": data.get("trade_id", data.index.astype(str)).astype(str),
            "symbol": data["symbol"].astype(str),
            "candidate": np.where(cic_type == "CIC1", "CIC1_beta_extreme", "CIC2_beta_broad"),
            "entry_time": entry.astype(str),
            "entry_month": entry.dt.strftime("%Y-%m"),
            "period": "live_counterfactual",
            "meta_router_training_split": "live_counterfactual",
            "state_cluster_id": "live_unknown",
            "cic_type": cic_type,
            "btc_state": data.get("btc_state_at_entry", data.get("btc_state_at_signal", pd.Series("", index=data.index))).astype(str),
            "market_impulse_density": pd.to_numeric(data.get("volume_impulse_density_at_entry"), errors="coerce"),
            "cluster_density": pd.to_numeric(data.get("cluster_impulse_density_at_entry"), errors="coerce"),
            "beta_strength": pd.to_numeric(data.get("beta_extension_score_at_signal"), errors="coerce"),
            "local_shock_strength": pd.to_numeric(data.get("local_volume_shock_strength_at_signal"), errors="coerce"),
            "ret_4h": np.nan,
            "ret_4h_percentile": pd.to_numeric(data.get("beta_extension_score_at_signal"), errors="coerce"),
            "symbol_volatility_percentile": np.nan,
            "burst_count_so_far": pd.to_numeric(data.get("burst_count_so_far"), errors="coerce"),
            "minutes_since_burst_start": pd.to_numeric(data.get("time_since_burst_start_so_far"), errors="coerce"),
            "same_timestamp_peer_count": entry.groupby(entry).transform("count").astype(float),
            "walkforward_state_novelty": np.nan,
            "novelty_bucket": "live_unknown",
            "actual_net20_later": pd.to_numeric(data.get("net_return_20bp"), errors="coerce"),
        }
    )
    return out


def _pre_entry_router_counterfactual_live(
    trades: pd.DataFrame,
    *,
    train_root: Path = V22_ROUTER_TRAIN_ROOT,
    cfg: V22BConfig = V22BConfig(),
) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "trade_id",
        "signal_id",
        "symbol",
        "candidate",
        "feature_time",
        "entry_time",
        "router_score_status",
        "logistic_no_trade_prob",
        "logistic_no_trade_prob_t70",
        "logistic_no_trade_prob_t75",
        "logistic_no_trade_prob_t80",
        "would_skip_t70",
        "would_skip_t75",
        "would_skip_t80",
        "actual_action",
        "actual_net20_later",
        "counterfactual_delta_t70",
        "counterfactual_delta_t75",
        "counterfactual_delta_t80",
        "router_train_events",
        "router_train_months",
        "router_train_max_entry_time",
        "missing_router_features",
        "notes",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    pool = _shadow_trade_pool(trades, _p2_combined_spec())
    live_features = _router_live_feature_frame(pool)
    if live_features.empty:
        return pd.DataFrame(columns=columns)

    training = _router_read_csv(train_root / "meta_router_feature_matrix.csv")
    if training.empty:
        out = live_features.copy()
        out["router_score_status"] = "missing_training_dataset"
        out["logistic_no_trade_prob"] = np.nan
        out["missing_router_features"] = ""
        out["notes"] = "counterfactual_only_no_live_action"
    else:
        training["entry_time"] = pd.to_datetime(training["entry_time"], utc=True, errors="coerce")
        min_live_entry = pd.to_datetime(live_features["entry_time"], utc=True, errors="coerce").min()
        history = training[training["entry_time"].lt(min_live_entry)].copy() if pd.notna(min_live_entry) else training.copy()
        feature_cols = _router_feature_columns(training)
        missing_cols = [col for col in feature_cols if col not in live_features.columns]
        for col in missing_cols:
            live_features[col] = np.nan
        if not _router_train_ready(history, cfg):
            out = live_features.copy()
            out["router_score_status"] = "insufficient_prior_training"
            out["logistic_no_trade_prob"] = np.nan
        else:
            model = _router_fit_logistic(history, feature_cols, cfg)
            out = live_features.copy()
            out["router_score_status"] = "scored_prior_only_logistic"
            out["logistic_no_trade_prob"] = _router_predict_logistic(model, live_features)
        out["missing_router_features"] = ",".join(missing_cols)
        out["notes"] = "counterfactual_only_no_live_action"
        out["router_train_events"] = int(len(history))
        out["router_train_months"] = int(history["entry_month"].nunique()) if "entry_month" in history.columns else 0
        max_train_time = history["entry_time"].max() if "entry_time" in history.columns and not history.empty else pd.NaT
        out["router_train_max_entry_time"] = "" if pd.isna(max_train_time) else str(max_train_time)

    for threshold in PRE_ENTRY_ROUTER_THRESHOLDS:
        suffix = int(round(threshold * 100))
        out[f"logistic_no_trade_prob_t{suffix}"] = out["logistic_no_trade_prob"]
        skip = pd.to_numeric(out["logistic_no_trade_prob"], errors="coerce").ge(threshold)
        out[f"would_skip_t{suffix}"] = skip.fillna(False)
        out[f"counterfactual_delta_t{suffix}"] = np.where(
            skip,
            -pd.to_numeric(out["actual_net20_later"], errors="coerce"),
            0.0,
        )
    out["candidate_id"] = out["signal_id"].astype(str)
    out["feature_time"] = out["entry_time"]
    out["actual_action"] = "p2_candidate_observed"
    if "router_train_events" not in out.columns:
        out["router_train_events"] = 0
    if "router_train_months" not in out.columns:
        out["router_train_months"] = 0
    if "router_train_max_entry_time" not in out.columns:
        out["router_train_max_entry_time"] = ""
    return out.reindex(columns=columns)


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


def _prepared_checkpoint_prices(prepared: pd.DataFrame) -> pd.DataFrame:
    if prepared.empty:
        return pd.DataFrame()
    wanted = ["exchange", "symbol", "feature_time", "bar_open_time", "open", "close"]
    cols = [col for col in wanted if col in prepared.columns]
    prices = prepared[cols].copy()
    prices["feature_time"] = pd.to_datetime(prices.get("feature_time"), utc=True, errors="coerce")
    if "bar_open_time" in prices.columns:
        prices["bar_open_time"] = pd.to_datetime(prices["bar_open_time"], utc=True, errors="coerce")
    else:
        prices["bar_open_time"] = prices["feature_time"]
    for col in ("open", "close"):
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    return prices.dropna(subset=["feature_time", "symbol", "open", "close"]).sort_values(["symbol", "feature_time"])


def _derive_net_at_cost(frame: pd.DataFrame, cost: int, price_col: str = "exit_price") -> pd.Series:
    entry = pd.to_numeric(frame.get("entry_price", pd.Series(dtype=float)), errors="coerce")
    price = pd.to_numeric(frame.get(price_col, pd.Series(dtype=float)), errors="coerce")
    return price / entry - 1.0 - 2.0 * float(cost) / 10_000.0


def _ensure_trade_exit_price(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    if "exit_price" not in out.columns:
        gross = pd.to_numeric(out.get("gross_return"), errors="coerce")
        entry = pd.to_numeric(out.get("entry_price"), errors="coerce")
        out["exit_price"] = entry * (1.0 + gross)
    return out


def _attach_checkpoint_prices(
    pool: pd.DataFrame,
    prepared: pd.DataFrame,
    *,
    checkpoint_minutes: int = CHECKPOINT_MINUTES,
) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    prices = _prepared_checkpoint_prices(prepared)
    out = _ensure_trade_exit_price(pool).copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["checkpoint_minutes"] = int(checkpoint_minutes)
    out["checkpoint_time"] = out["entry_time"] + pd.Timedelta(minutes=int(checkpoint_minutes))
    rows: list[dict[str, object]] = []
    for symbol, group in out.groupby("symbol", sort=False):
        symbol_prices = prices[prices["symbol"].astype(str).eq(str(symbol))].sort_values("feature_time")
        for row in group.sort_values("checkpoint_time").itertuples(index=False):
            payload = row._asdict()
            checkpoint = pd.Timestamp(payload["checkpoint_time"])
            eligible = symbol_prices[symbol_prices["feature_time"].le(checkpoint)]
            next_bar = symbol_prices[symbol_prices["feature_time"].gt(checkpoint)]
            if eligible.empty:
                payload["checkpoint_snapshot_time"] = pd.NaT
                payload["checkpoint_price"] = np.nan
            else:
                last = eligible.iloc[-1]
                payload["checkpoint_snapshot_time"] = last["feature_time"]
                payload["checkpoint_price"] = float(last["close"])
            if next_bar.empty:
                payload["checkpoint_exit_time"] = pd.NaT
                payload["checkpoint_exit_price"] = np.nan
            else:
                nxt = next_bar.iloc[0]
                exit_time = nxt["bar_open_time"]
                if pd.isna(exit_time) or pd.Timestamp(exit_time) < checkpoint:
                    exit_time = nxt["feature_time"]
                payload["checkpoint_exit_time"] = exit_time
                payload["checkpoint_exit_price"] = float(nxt["open"])
            rows.append(payload)
    checked = pd.DataFrame(rows)
    if checked.empty:
        return out
    checked["checkpoint_price_covered"] = checked["checkpoint_price"].notna()
    checked["checkpoint_exit_covered"] = checked["checkpoint_exit_price"].notna()
    checked["checkpoint_execution_granularity"] = "15m_checkpoint_execution"
    for cost in (10, 20, 30):
        checked[f"checkpoint_net{cost}"] = (
            pd.to_numeric(checked["checkpoint_price"], errors="coerce")
            / pd.to_numeric(checked["entry_price"], errors="coerce")
            - 1.0
            - 2.0 * float(cost) / 10_000.0
        )
        checked[f"net_if_checkpoint_exit_{cost}bp"] = _derive_net_at_cost(
            checked.rename(columns={"checkpoint_exit_price": "_checkpoint_exit_price"}),
            cost,
            "_checkpoint_exit_price",
        )
    return checked.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _checkpoint_rule(spec: dict[str, object]) -> str:
    if "checkpoint_rule" in spec:
        return str(spec["checkpoint_rule"])
    return "cp60_all" if bool(spec.get("checkpoint_enabled", False)) else "none"


def _beta_high_score(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in (
        "c2_beta_extension_score",
        "beta_extension_score_at_entry",
        "beta_extension_score_at_signal",
        "rank_beta_extreme_strength",
        "ret_4h_percentile",
    ):
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _protect_a_cap2_mask(out: pd.DataFrame, raw_trigger: pd.Series) -> pd.Series:
    beta_score = _beta_high_score(out)
    beta_high = beta_score.ge(CP60_PROTECT_A_BETA_HIGH_THRESHOLD)
    candidates = out[raw_trigger.fillna(False).astype(bool) & beta_high.fillna(False)].copy()
    protect = pd.Series(False, index=out.index)
    if candidates.empty:
        return protect
    candidates["_checkpoint_sort"] = pd.to_datetime(candidates["checkpoint_time"], utc=True, errors="coerce")
    candidates["_entry_sort"] = pd.to_datetime(candidates["entry_time"], utc=True, errors="coerce")
    sort_cols = ["burst_id", "_checkpoint_sort", "_entry_sort", "symbol"] if "burst_id" in candidates.columns else ["_checkpoint_sort", "_entry_sort", "symbol"]
    candidates = candidates.sort_values(sort_cols)
    group_key = "burst_id" if "burst_id" in candidates.columns else pd.Series("unknown", index=candidates.index)
    keep_idx = candidates.groupby(group_key, sort=False, dropna=False).head(CP60_PROTECT_A_CAP_PER_BURST).index
    protect.loc[keep_idx] = True
    return protect


def _apply_cp60_effective_exits(pool: pd.DataFrame, *, checkpoint_enabled: bool, checkpoint_rule: str | None = None) -> pd.DataFrame:
    out = _ensure_trade_exit_price(pool).copy()
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["checkpoint_time"] = pd.to_datetime(out.get("checkpoint_time"), utc=True, errors="coerce")
    checkpoint_before_exit = out["checkpoint_time"].lt(out["exit_time"])
    rule = checkpoint_rule or ("cp60_all" if checkpoint_enabled else "none")
    raw_trigger = (
        bool(checkpoint_enabled)
        & out.get("checkpoint_price_covered", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        & out.get("checkpoint_exit_covered", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        & checkpoint_before_exit
        & pd.to_numeric(out.get("checkpoint_net20"), errors="coerce").le(0.0)
    )
    if isinstance(raw_trigger, bool):
        raw_trigger = pd.Series(False, index=out.index)
    raw_trigger = raw_trigger.fillna(False).astype(bool)
    protected = pd.Series(False, index=out.index)
    if rule == "cp60_protect_a_cap2":
        protected = _protect_a_cap2_mask(out, raw_trigger)
    elif rule not in {"none", "cp60_all"}:
        raise ValueError(f"unsupported checkpoint rule: {rule}")
    trigger = raw_trigger & ~protected
    beta_score = _beta_high_score(out)
    out["checkpoint_enabled"] = bool(checkpoint_enabled)
    out["checkpoint_rule"] = rule
    out["cp60_would_exit"] = raw_trigger
    out["beta_high_protection"] = beta_score.ge(CP60_PROTECT_A_BETA_HIGH_THRESHOLD).fillna(False)
    out["beta_high_threshold"] = CP60_PROTECT_A_BETA_HIGH_THRESHOLD
    out["beta_high_score"] = beta_score
    out["protected_by_beta_high"] = protected.fillna(False).astype(bool)
    out["protection_cap"] = CP60_PROTECT_A_CAP_PER_BURST if rule == "cp60_protect_a_cap2" else 0
    out["checkpoint_triggered"] = trigger.fillna(False).astype(bool)
    if "burst_id" in out.columns:
        protected_order = (
            out[out["protected_by_beta_high"]]
            .sort_values(["burst_id", "checkpoint_time", "entry_time", "symbol"])
            .groupby("burst_id", sort=False)
            .cumcount()
        )
        out["protected_burst_count_before"] = np.nan
        out.loc[protected_order.index, "protected_burst_count_before"] = protected_order.astype(float)
        out["protected_burst_count_after"] = out["protected_burst_count_before"] + 1.0
    else:
        out["protected_burst_count_before"] = np.nan
        out["protected_burst_count_after"] = np.nan
    out["original_exit_time"] = out["exit_time"]
    out["original_exit_price"] = out["exit_price"]
    out["net_if_kept_counterfactual_10bp"] = pd.to_numeric(out.get("net_return_10bp"), errors="coerce")
    out["net_if_kept_counterfactual_20bp"] = pd.to_numeric(out.get("net_return_20bp"), errors="coerce")
    out["net_if_kept_counterfactual_30bp"] = pd.to_numeric(out.get("net_return_30bp"), errors="coerce")
    out["counterfactual_cp60_exit_net20"] = pd.to_numeric(out.get("net_if_checkpoint_exit_20bp"), errors="coerce")
    out["actual_keep_exit_net20"] = pd.to_numeric(out.get("net_return_20bp"), errors="coerce")
    out["delta_vs_cp60"] = out["actual_keep_exit_net20"] - out["counterfactual_cp60_exit_net20"]
    out["slot_blocked_minutes"] = (
        pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
        - pd.to_datetime(out["checkpoint_exit_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    out.loc[~out["protected_by_beta_high"], "slot_blocked_minutes"] = np.nan
    out["missed_trade_due_to_protection"] = False
    out["effective_exit_time"] = out["exit_time"]
    out["effective_exit_price"] = out["exit_price"]
    out.loc[out["checkpoint_triggered"], "effective_exit_time"] = out.loc[out["checkpoint_triggered"], "checkpoint_exit_time"]
    out.loc[out["checkpoint_triggered"], "effective_exit_price"] = out.loc[out["checkpoint_triggered"], "checkpoint_exit_price"]
    for cost in (10, 20, 30):
        original = pd.to_numeric(out.get(_net_col(cost)), errors="coerce")
        cp = pd.to_numeric(out.get(f"net_if_checkpoint_exit_{cost}bp"), errors="coerce")
        out[f"effective_net_return_{cost}bp"] = original
        out.loc[out["checkpoint_triggered"], f"effective_net_return_{cost}bp"] = cp
    return out


def _checkpoint_shadow_specs() -> list[dict[str, object]]:
    return [
        {
            "portfolio_id": CHECKPOINT_PORTFOLIO_IDS["S0"],
            "label": "S0",
            "checkpoint_enabled": False,
            "checkpoint_rule": "none",
            "overflow_enabled": False,
            "description": "P2 CIC1+CIC2 max8 baseline.",
        },
        {
            "portfolio_id": CHECKPOINT_PORTFOLIO_IDS["S1"],
            "label": "S1",
            "checkpoint_enabled": False,
            "checkpoint_rule": "none",
            "overflow_enabled": True,
            "description": "P2 max8 plus O6 late-burst overflow.",
        },
        {
            "portfolio_id": CHECKPOINT_PORTFOLIO_IDS["S2"],
            "label": "S2",
            "checkpoint_enabled": True,
            "checkpoint_rule": "cp60_all",
            "overflow_enabled": False,
            "description": "P2 max8 with CP60 no-follow-through checkpoint.",
        },
        {
            "portfolio_id": CHECKPOINT_PORTFOLIO_IDS["S3"],
            "label": "S3",
            "checkpoint_enabled": True,
            "checkpoint_rule": "cp60_all",
            "overflow_enabled": True,
            "description": "P2 max8 with CP60 plus O6 late-burst overflow.",
        },
        {
            "portfolio_id": CHECKPOINT_PORTFOLIO_IDS["S4"],
            "label": "S4",
            "checkpoint_enabled": True,
            "checkpoint_rule": "cp60_protect_a_cap2",
            "overflow_enabled": False,
            "description": "P2 max8 with CP60 Protect_A beta_high cap2 shadow.",
        },
        {
            "portfolio_id": CHECKPOINT_PORTFOLIO_IDS["S5"],
            "label": "S5",
            "checkpoint_enabled": True,
            "checkpoint_rule": "cp60_protect_a_cap2",
            "overflow_enabled": True,
            "description": "P2 max8 with CP60 Protect_A beta_high cap2 plus O6 shadow.",
        },
    ]


def _checkpoint_payload(row: Any, spec: dict[str, object], *, concurrent: int, core_count: int, overflow_count: int) -> dict[str, object]:
    payload = row._asdict()
    payload["portfolio_id"] = spec["portfolio_id"]
    payload["portfolio_label"] = spec["label"]
    payload["portfolio_description"] = spec["description"]
    payload["checkpoint_rule"] = _checkpoint_rule(spec)
    payload["checkpoint_main_cost_bps"] = CHECKPOINT_TRIGGER_COST_BPS
    payload["concurrent_positions"] = concurrent
    payload["core_positions_at_time"] = core_count
    payload["overflow_positions_at_time"] = overflow_count
    payload["max_positions_at_time"] = 8
    payload["is_core"] = True
    payload["is_overflow"] = False
    payload["sleeve"] = "core"
    payload["position_size"] = 1.0
    payload["extra_exposure"] = 0.0
    payload["overflow_reason"] = ""
    payload["overflow_slot_index"] = np.nan
    payload["selected"] = True
    payload["skip_reason"] = ""
    return payload


def _checkpoint_overflow_size(row: Any) -> float:
    candidate = str(getattr(row, "candidate", ""))
    if candidate == "CIC1_FILTERED_MIR1":
        return 0.50
    if candidate == "CIC2_FILTERED_MIR1":
        return 0.25
    return 0.0


def _select_checkpoint_shadow_portfolio(pool: pd.DataFrame, spec: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pool.copy(), pool.copy()
    data = _add_shadow_burst_phase(pool)
    data = _apply_cp60_effective_exits(
        data,
        checkpoint_enabled=bool(spec["checkpoint_enabled"]),
        checkpoint_rule=_checkpoint_rule(spec),
    )
    data["rank_score"] = 0.0
    data["rank_position"] = data.groupby("entry_time", sort=False)["rank_score"].rank(ascending=False, method="first").astype(int)
    data = data.sort_values(["entry_time", "symbol", "candidate_priority"], ascending=[True, True, False])
    active_core: list[tuple[pd.Timestamp, str]] = []
    active_overflow: list[tuple[pd.Timestamp, str]] = []
    selected_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    overflow_enabled = bool(spec["overflow_enabled"])
    for row in data.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        active_core = [(exit_time, symbol) for exit_time, symbol in active_core if exit_time > entry]
        active_overflow = [(exit_time, symbol) for exit_time, symbol in active_overflow if exit_time > entry]
        active_symbols = {symbol for _, symbol in [*active_core, *active_overflow]}
        payload = _checkpoint_payload(
            row,
            spec,
            concurrent=len(active_core) + len(active_overflow),
            core_count=len(active_core),
            overflow_count=len(active_overflow),
        )
        if str(row.symbol) in active_symbols:
            payload["selected"] = False
            payload["skip_reason"] = "symbol_already_active"
            skipped_rows.append(payload)
            continue
        if len(active_core) < 8:
            selected_rows.append(payload)
            active_core.append((pd.Timestamp(row.effective_exit_time), str(row.symbol)))
            continue
        overflow_size = _checkpoint_overflow_size(row)
        overflow_eligible = overflow_enabled and int(getattr(row, "burst_count_so_far", 0)) >= 9 and overflow_size > 0
        if overflow_eligible and len(active_overflow) < 4:
            payload["is_core"] = False
            payload["is_overflow"] = True
            payload["sleeve"] = "overflow"
            payload["position_size"] = overflow_size
            payload["extra_exposure"] = overflow_size
            payload["overflow_reason"] = "portfolio_full_late_burst_overflow"
            payload["overflow_slot_index"] = len(active_overflow) + 1
            selected_rows.append(payload)
            active_overflow.append((pd.Timestamp(row.effective_exit_time), str(row.symbol)))
            continue
        payload["selected"] = False
        payload["skip_reason"] = (
            "overflow_full" if overflow_eligible else "portfolio_full_not_overflow_eligible"
        )
        skipped_rows.append(payload)
    return pd.DataFrame(selected_rows), pd.DataFrame(skipped_rows)


def _checkpoint_trade_pool(trades: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    spec = {
        "candidates": ["CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"],
        "pool": "CIC1_CIC2_COMBINED",
    }
    pool = _shadow_trade_pool(trades, spec)
    if pool.empty:
        return pool
    return _attach_checkpoint_prices(pool, prepared, checkpoint_minutes=CHECKPOINT_MINUTES)


def checkpoint_shadow_live(
    trades: pd.DataFrame,
    prepared: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _checkpoint_trade_pool(trades, prepared)
    ledgers: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for spec in _checkpoint_shadow_specs():
        selected, skipped = _select_checkpoint_shadow_portfolio(pool, spec)
        for frame, selected_flag in ((selected, True), (skipped, False)):
            if frame.empty:
                continue
            frame = frame.copy()
            frame["selected"] = selected_flag
            for cost in (10, 20, 30):
                frame[f"net{cost}_if_taken"] = pd.to_numeric(frame.get(f"effective_net_return_{cost}bp"), errors="coerce")
            if selected_flag:
                ledgers.append(frame)
            else:
                skipped_frames.append(frame)
        for cost in (10, 20, 30):
            summary_rows.append(_checkpoint_summary_row(str(spec["portfolio_id"]), selected, skipped, cost=cost))
        date_sources = [
            frame["entry_time"]
            for frame in (selected, skipped)
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
            row = _checkpoint_summary_row(str(spec["portfolio_id"]), sel_day, skip_day, cost=20)
            row["date"] = day
            daily_rows.append(row)
        early_exits = int(selected.get("checkpoint_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not selected.empty else 0
        protected_exits = int(selected.get("protected_by_beta_high", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not selected.empty else 0
        cp60_would_exits = int(selected.get("cp60_would_exit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not selected.empty else 0
        eval_count = protected_exits if _checkpoint_rule(spec) == "cp60_protect_a_cap2" else early_exits
        if eval_count < 10 and _checkpoint_rule(spec) == "cp60_protect_a_cap2":
            sample_status = "system_check_only"
            evaluation_status = "no_decision"
        elif eval_count < 20:
            sample_status = "system_check_only"
            evaluation_status = "no_decision"
        elif eval_count < 50:
            sample_status = "behavior_check"
            evaluation_status = "no_upgrade_no_downgrade"
        elif eval_count < 100:
            sample_status = "initial_checkpoint_evaluation"
            evaluation_status = "evaluate_shadow_only"
        else:
            sample_status = "candidate_check"
            evaluation_status = "consider_shadow_upgrade"
        status_rows.append(
            {
                "portfolio_id": spec["portfolio_id"],
                "label": spec["label"],
                "checkpoint_enabled": bool(spec["checkpoint_enabled"]),
                "overflow_enabled": bool(spec["overflow_enabled"]),
                "selected_trades": int(len(selected)),
                "skipped_candidates": int(len(skipped)),
                "checkpoint_exits": early_exits,
                "cp60_would_exits": cp60_would_exits,
                "protected_exits": protected_exits,
                "protection_cap": CP60_PROTECT_A_CAP_PER_BURST
                if _checkpoint_rule(spec) == "cp60_protect_a_cap2"
                else 0,
                "beta_high_threshold": CP60_PROTECT_A_BETA_HIGH_THRESHOLD
                if _checkpoint_rule(spec) == "cp60_protect_a_cap2"
                else np.nan,
                "sample_status": sample_status,
                "evaluation_status": evaluation_status,
                "real_live_allowed": False,
            }
        )
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    daily = pd.DataFrame(daily_rows)
    status = pd.DataFrame(status_rows)
    slot = _slot_release_attribution_live(ledger)
    return status, ledger, skipped_all, daily, summary, slot


def _effective_net_col(cost: int) -> str:
    return f"effective_net_return_{cost}bp"


def _checkpoint_summary_row(portfolio_id: str, selected: pd.DataFrame, skipped: pd.DataFrame, *, cost: int) -> dict[str, object]:
    if cost == 50:
        selected_net = pd.to_numeric(selected.get("net_return_50bp", pd.Series(dtype=float)), errors="coerce")
        skipped_net = pd.to_numeric(skipped.get("net_return_50bp", pd.Series(dtype=float)), errors="coerce")
    else:
        selected_net = pd.to_numeric(selected.get(_effective_net_col(cost), pd.Series(dtype=float)), errors="coerce")
        skipped_net = pd.to_numeric(skipped.get(_effective_net_col(cost), pd.Series(dtype=float)), errors="coerce")
    weights = pd.to_numeric(selected.get("position_size", pd.Series(1.0, index=selected.index)), errors="coerce").fillna(1.0)
    weighted_sum = float((selected_net * weights).sum()) if len(selected_net) else 0.0
    core = selected[selected.get("is_core", pd.Series(dtype=bool)).fillna(False).astype(bool)] if not selected.empty else selected
    overflow = selected[selected.get("is_overflow", pd.Series(dtype=bool)).fillna(False).astype(bool)] if not selected.empty else selected
    overflow_weights = pd.to_numeric(overflow.get("position_size", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    overflow_net = pd.to_numeric(overflow.get(_effective_net_col(cost), pd.Series(dtype=float)), errors="coerce")
    early_exits = int(selected.get("checkpoint_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not selected.empty else 0
    cp60_would_exits = int(selected.get("cp60_would_exit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not selected.empty else 0
    protected_exits = int(selected.get("protected_by_beta_high", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not selected.empty else 0
    protected_delta = pd.to_numeric(
        selected.loc[selected.get("protected_by_beta_high", pd.Series(False, index=selected.index)).fillna(False).astype(bool), "delta_vs_cp60"]
        if not selected.empty and "delta_vs_cp60" in selected.columns
        else pd.Series(dtype=float),
        errors="coerce",
    )
    return {
        "portfolio_id": portfolio_id,
        "cost_single_side_bps": cost,
        "selected_trades": int(len(selected)),
        "core_trades": int(len(core)),
        "overflow_trades": int(len(overflow)),
        "skipped_candidates": int(len(skipped)),
        "checkpoint_exits": early_exits,
        "cp60_would_exits": cp60_would_exits,
        "protected_exits": protected_exits,
        "protected_delta_vs_cp60_sum": _safe_float(protected_delta.sum()) if len(protected_delta) else 0.0,
        "protected_delta_vs_cp60_avg": _safe_float(protected_delta.mean()) if len(protected_delta) else np.nan,
        "selected_effective_net": _safe_float(selected_net.mean()),
        "skipped_effective_net": _safe_float(skipped_net.mean()),
        "selected_minus_skipped": _safe_float(selected_net.mean() - skipped_net.mean())
        if len(selected_net) and len(skipped_net)
        else np.nan,
        "portfolio_net": weighted_sum / 8.0,
        "core_net": _safe_float(pd.to_numeric(core.get(_effective_net_col(cost), pd.Series(dtype=float)), errors="coerce").mean()),
        "overflow_net": _safe_float((overflow_net * overflow_weights).sum() / overflow_weights.sum())
        if float(overflow_weights.sum()) > 0
        else np.nan,
        "extra_exposure": float(overflow_weights.sum()),
        "incremental_return_per_extra_exposure": float((overflow_net * overflow_weights).sum() / overflow_weights.sum())
        if float(overflow_weights.sum()) > 0
        else np.nan,
        "real_live_allowed": False,
    }


def _slot_release_attribution_live(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    base_by_label = {
        "S2": set(ledger[ledger["portfolio_id"].astype(str).eq(CHECKPOINT_PORTFOLIO_IDS["S0"])]["signal_id"].astype(str)),
        "S3": set(ledger[ledger["portfolio_id"].astype(str).eq(CHECKPOINT_PORTFOLIO_IDS["S1"])]["signal_id"].astype(str)),
        "S4": set(ledger[ledger["portfolio_id"].astype(str).eq(CHECKPOINT_PORTFOLIO_IDS["S0"])]["signal_id"].astype(str)),
        "S5": set(ledger[ledger["portfolio_id"].astype(str).eq(CHECKPOINT_PORTFOLIO_IDS["S1"])]["signal_id"].astype(str)),
    }
    rows: list[dict[str, object]] = []
    cp_ids = [
        CHECKPOINT_PORTFOLIO_IDS["S2"],
        CHECKPOINT_PORTFOLIO_IDS["S3"],
        CHECKPOINT_PORTFOLIO_IDS["S4"],
        CHECKPOINT_PORTFOLIO_IDS["S5"],
    ]
    cp = ledger[ledger["portfolio_id"].astype(str).isin(cp_ids)].copy()
    for row in cp.itertuples(index=False):
        label = str(getattr(row, "portfolio_label", ""))
        signal_id = str(getattr(row, "signal_id", ""))
        new_trade = signal_id not in base_by_label.get(label, set())
        checkpoint_triggered = bool(getattr(row, "checkpoint_triggered", False))
        net_if_kept = getattr(row, "net_if_kept_counterfactual_20bp", np.nan)
        net_if_exited = getattr(row, "effective_net_return_20bp", np.nan)
        rows.append(
            {
                "portfolio_id": getattr(row, "portfolio_id", ""),
                "trade_id": getattr(row, "trade_id", ""),
                "signal_id": signal_id,
                "symbol": getattr(row, "symbol", ""),
                "candidate": getattr(row, "candidate", ""),
                "entry_time": getattr(row, "entry_time", pd.NaT),
                "checkpoint_time": getattr(row, "checkpoint_time", pd.NaT),
                "checkpoint_net20": getattr(row, "checkpoint_net20", np.nan),
                "exit_by_checkpoint": checkpoint_triggered,
                "cp60_would_exit": bool(getattr(row, "cp60_would_exit", False)),
                "protected_by_beta_high": bool(getattr(row, "protected_by_beta_high", False)),
                "net_if_kept_counterfactual": net_if_kept,
                "net_if_checkpoint_exit": net_if_exited if checkpoint_triggered else np.nan,
                "slot_released": checkpoint_triggered,
                "new_trade_entered_due_to_release": new_trade,
                "new_trade_net20": net_if_exited if new_trade else np.nan,
                "avoidance_pnl": (net_if_exited - net_if_kept) if checkpoint_triggered else 0.0,
                "opportunity_capture": net_if_exited if new_trade else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _checkpoint_protection_attribution_live(ledger: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "portfolio_id",
        "comparison_cp60_portfolio_id",
        "trade_id",
        "signal_id",
        "symbol",
        "candidate",
        "entry_time",
        "checkpoint_time",
        "checkpoint_exit_time",
        "original_exit_time",
        "checkpoint_net20",
        "counterfactual_cp60_exit_net20",
        "actual_keep_exit_net20",
        "delta_vs_cp60",
        "beta_high_score",
        "beta_high_threshold",
        "protected_burst_count_before",
        "protected_burst_count_after",
        "protection_cap",
        "slot_blocked_minutes",
        "missed_trade_due_to_protection",
        "missed_trade_count",
        "missed_trade_ids",
        "missed_trade_net20_sum",
        "total_effect_after_missed",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    pairs = {
        CHECKPOINT_PORTFOLIO_IDS["S4"]: CHECKPOINT_PORTFOLIO_IDS["S2"],
        CHECKPOINT_PORTFOLIO_IDS["S5"]: CHECKPOINT_PORTFOLIO_IDS["S3"],
    }
    rows: list[dict[str, object]] = []
    for protect_id, cp_id in pairs.items():
        protected_ledger = ledger[ledger["portfolio_id"].astype(str).eq(protect_id)].copy()
        cp_ledger = ledger[ledger["portfolio_id"].astype(str).eq(cp_id)].copy()
        if protected_ledger.empty:
            continue
        protected_keys = set(protected_ledger["signal_id"].astype(str))
        cp_only = cp_ledger[~cp_ledger["signal_id"].astype(str).isin(protected_keys)].copy() if not cp_ledger.empty else pd.DataFrame()
        protected_rows = protected_ledger[
            protected_ledger.get("protected_by_beta_high", pd.Series(False, index=protected_ledger.index)).fillna(False).astype(bool)
        ].copy()
        for row in protected_rows.itertuples(index=False):
            checkpoint_exit_time = pd.Timestamp(getattr(row, "checkpoint_exit_time", pd.NaT))
            original_exit_time = pd.Timestamp(getattr(row, "original_exit_time", getattr(row, "exit_time", pd.NaT)))
            missed = pd.DataFrame()
            if not cp_only.empty and pd.notna(checkpoint_exit_time) and pd.notna(original_exit_time):
                entry = pd.to_datetime(cp_only["entry_time"], utc=True, errors="coerce")
                missed = cp_only[entry.ge(checkpoint_exit_time) & entry.lt(original_exit_time)].copy()
            missed_net = pd.to_numeric(missed.get("effective_net_return_20bp", pd.Series(dtype=float)), errors="coerce")
            delta_vs_cp60 = float(pd.to_numeric(pd.Series([getattr(row, "delta_vs_cp60", np.nan)]), errors="coerce").iloc[0])
            rows.append(
                {
                    "portfolio_id": protect_id,
                    "comparison_cp60_portfolio_id": cp_id,
                    "trade_id": getattr(row, "trade_id", ""),
                    "signal_id": getattr(row, "signal_id", ""),
                    "symbol": getattr(row, "symbol", ""),
                    "candidate": getattr(row, "candidate", ""),
                    "entry_time": getattr(row, "entry_time", pd.NaT),
                    "checkpoint_time": getattr(row, "checkpoint_time", pd.NaT),
                    "checkpoint_exit_time": getattr(row, "checkpoint_exit_time", pd.NaT),
                    "original_exit_time": getattr(row, "original_exit_time", pd.NaT),
                    "checkpoint_net20": getattr(row, "checkpoint_net20", np.nan),
                    "counterfactual_cp60_exit_net20": getattr(row, "counterfactual_cp60_exit_net20", np.nan),
                    "actual_keep_exit_net20": getattr(row, "actual_keep_exit_net20", np.nan),
                    "delta_vs_cp60": delta_vs_cp60,
                    "beta_high_score": getattr(row, "beta_high_score", np.nan),
                    "beta_high_threshold": getattr(row, "beta_high_threshold", CP60_PROTECT_A_BETA_HIGH_THRESHOLD),
                    "protected_burst_count_before": getattr(row, "protected_burst_count_before", np.nan),
                    "protected_burst_count_after": getattr(row, "protected_burst_count_after", np.nan),
                    "protection_cap": getattr(row, "protection_cap", CP60_PROTECT_A_CAP_PER_BURST),
                    "slot_blocked_minutes": getattr(row, "slot_blocked_minutes", np.nan),
                    "missed_trade_due_to_protection": not missed.empty,
                    "missed_trade_count": int(len(missed)),
                    "missed_trade_ids": ",".join(missed["signal_id"].astype(str).tolist()) if not missed.empty else "",
                    "missed_trade_net20_sum": float(missed_net.sum()) if len(missed_net) else 0.0,
                    "total_effect_after_missed": delta_vs_cp60 - (float(missed_net.sum()) if len(missed_net) else 0.0),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _checkpoint_daily_summary(ledger: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return daily
    out = daily.copy()
    if out.empty:
        return out
    return out.sort_values(["date", "portfolio_id"]).reset_index(drop=True)


def _checkpoint_local_time_audit(pool: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for minutes in (45, 60, 75):
        checked = _attach_checkpoint_prices(pool, prepared, checkpoint_minutes=minutes)
        selected, skipped = _select_checkpoint_shadow_portfolio(
            checked,
            {
                "portfolio_id": f"P2_MAX8_CP{minutes}",
                "label": f"CP{minutes}",
                "checkpoint_enabled": True,
                "overflow_enabled": False,
                "description": f"P2 max8 CP{minutes} local audit.",
            },
        )
        row = _checkpoint_summary_row(f"P2_MAX8_CP{minutes}", selected, skipped, cost=20)
        row["checkpoint_minutes"] = minutes
        rows.append(row)
    return pd.DataFrame(rows)


def _checkpoint_candidate_type_audit(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    data = ledger[ledger["portfolio_id"].astype(str).eq(CHECKPOINT_PORTFOLIO_IDS["S2"])].copy()
    rows: list[dict[str, object]] = []
    for candidate, group in data.groupby("candidate", sort=False, dropna=False):
        rows.append(
            {
                "candidate": candidate,
                "trades": int(len(group)),
                "checkpoint_exits": int(group.get("checkpoint_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "effective_net20": _safe_float(pd.to_numeric(group.get("effective_net_return_20bp"), errors="coerce").mean()),
                "kept_counterfactual_net20": _safe_float(pd.to_numeric(group.get("net_if_kept_counterfactual_20bp"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_checkpoint_status(
    report_root: Path,
    status: pd.DataFrame,
    summary: pd.DataFrame,
    slot: pd.DataFrame,
) -> None:
    lines = [
        "# v1.3G Checkpoint Shadow Live",
        "",
        "- status: shadow_only",
        "- real_live_allowed: false",
        "- checkpoint_rule: entry_time + 60m, latest 15m close <= checkpoint_time, exit next 15m open when net20 <= 0",
        "- protect_a_cap2_rule: if CP60 would exit and beta_high_score >= frozen threshold, protect at most 2 exits per burst",
        f"- protect_a_beta_high_threshold: {CP60_PROTECT_A_BETA_HIGH_THRESHOLD}",
        "- primary_replacement_allowed: false",
        "",
        "## Portfolios",
    ]
    if status.empty:
        lines.append("- No checkpoint shadow rows yet.")
    else:
        for row in status.itertuples(index=False):
            lines.append(
                f"- {row.portfolio_id}: selected={row.selected_trades}, skipped={row.skipped_candidates}, "
                f"checkpoint_exits={row.checkpoint_exits}, protected_exits={getattr(row, 'protected_exits', 0)}, "
                f"sample_status={row.sample_status}, "
                f"evaluation_status={row.evaluation_status}"
            )
    focal = summary[
        pd.to_numeric(summary.get("cost_single_side_bps"), errors="coerce").eq(20)
    ] if not summary.empty else pd.DataFrame()
    if not focal.empty:
        lines.extend(["", "## 20bp Comparison"])
        for row in focal.itertuples(index=False):
            lines.append(
                f"- {row.portfolio_id}: portfolio_net20={row.portfolio_net:.4%}, "
                f"selected_net20={row.selected_effective_net:.4%}, skipped_net20={row.skipped_effective_net:.4%}, "
                f"protected={getattr(row, 'protected_exits', 0)}, delta={row.selected_minus_skipped:.4%}"
            )
    if not slot.empty:
        triggered = slot[slot.get("exit_by_checkpoint", pd.Series(dtype=bool)).fillna(False).astype(bool)]
        new_trades = slot[slot.get("new_trade_entered_due_to_release", pd.Series(dtype=bool)).fillna(False).astype(bool)]
        lines.extend(
            [
                "",
                "## Slot Release",
                f"- checkpoint_exit_rows: {len(triggered)}",
                f"- new_trades_due_to_release: {len(new_trades)}",
                f"- avoidance_pnl_sum: {pd.to_numeric(triggered.get('avoidance_pnl'), errors='coerce').sum():.4%}",
                f"- opportunity_capture_sum: {pd.to_numeric(new_trades.get('opportunity_capture'), errors='coerce').sum():.4%}",
            ]
        )
    (report_root / "checkpoint_current_status.md").write_text("\n".join(lines), encoding="utf-8")


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


def _token_attention_counterfactual_live(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    try:
        from pressure_graph.reports.v67_token_attention_forward_context import (
            build_token_attention_context_for_trades,
        )

        return build_token_attention_context_for_trades(trades)
    except Exception as exc:  # noqa: BLE001 - context logging must never break paper-live
        return pd.DataFrame(
            [
                {
                    "trade_id": "",
                    "token_attention_context_error": str(exc),
                    "live_action_allowed": False,
                    "recommended_use": "forward_counterfactual_diagnostic_only",
                }
            ]
        )


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
    checkpoint_status, checkpoint_ledger, checkpoint_skipped, checkpoint_daily, checkpoint_summary, slot_release = (
        checkpoint_shadow_live(trades, prepared)
    )
    checkpoint_pool = _checkpoint_trade_pool(trades, prepared)
    checkpoint_time_audit = _checkpoint_local_time_audit(checkpoint_pool, prepared)
    checkpoint_candidate_type_audit = _checkpoint_candidate_type_audit(checkpoint_ledger)
    checkpoint_protection_attribution = _checkpoint_protection_attribution_live(checkpoint_ledger)
    pre_entry_router_counterfactual = _pre_entry_router_counterfactual_live(trades)
    token_attention_counterfactual = _token_attention_counterfactual_live(trades)
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
        "checkpoint_trade_ledger": report_root / "checkpoint_trade_ledger.parquet",
        "checkpoint_skipped_candidates": report_root / "checkpoint_skipped_candidates.parquet",
        "checkpoint_daily_summary": report_root / "checkpoint_daily_summary.csv",
        "checkpoint_comparison": report_root / "checkpoint_comparison.csv",
        "slot_release_attribution_live": report_root / "slot_release_attribution_live.csv",
        "checkpoint_protection_attribution_live": report_root / "checkpoint_protection_attribution_live.csv",
        "checkpoint_current_status": report_root / "checkpoint_current_status.md",
        "checkpoint_45_60_75_audit": report_root / "checkpoint_45_60_75_audit.csv",
        "checkpoint_candidate_type_audit": report_root / "checkpoint_candidate_type_audit.csv",
        "pre_entry_router_counterfactual_live": report_root / "pre_entry_router_counterfactual_live.csv",
        "pre_entry_router_counterfactual_live_data": report_root / "pre_entry_router_counterfactual_live.parquet",
        "token_attention_counterfactual_live": report_root / "token_attention_counterfactual_live.csv",
        "token_attention_counterfactual_live_data": report_root / "token_attention_counterfactual_live.parquet",
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
    write_parquet(checkpoint_ledger, outputs["checkpoint_trade_ledger"])
    write_parquet(checkpoint_skipped, outputs["checkpoint_skipped_candidates"])
    checkpoint_daily.to_csv(outputs["checkpoint_daily_summary"], index=False)
    checkpoint_summary.to_csv(outputs["checkpoint_comparison"], index=False)
    slot_release.to_csv(outputs["slot_release_attribution_live"], index=False)
    checkpoint_protection_attribution.to_csv(outputs["checkpoint_protection_attribution_live"], index=False)
    checkpoint_time_audit.to_csv(outputs["checkpoint_45_60_75_audit"], index=False)
    checkpoint_candidate_type_audit.to_csv(outputs["checkpoint_candidate_type_audit"], index=False)
    pre_entry_router_counterfactual.to_csv(outputs["pre_entry_router_counterfactual_live"], index=False)
    write_parquet(pre_entry_router_counterfactual, outputs["pre_entry_router_counterfactual_live_data"])
    token_attention_counterfactual.to_csv(outputs["token_attention_counterfactual_live"], index=False)
    write_parquet(token_attention_counterfactual, outputs["token_attention_counterfactual_live_data"])
    _write_shadow_status(report_root, shadow_status, shadow_summary)
    _write_overflow_status(report_root, overflow_ledger, overflow_daily)
    _write_checkpoint_status(report_root, checkpoint_status, checkpoint_summary, slot_release)
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
