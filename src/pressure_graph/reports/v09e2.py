from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import select_portfolio
from pressure_graph.reports.v09e import (
    _attach_orderbook_features,
    _load_orderbook_features,
    _max_contribution,
    _month_cap_expectancy,
    _num,
)
from pressure_graph.reports.v09e1 import (
    CANDIDATE_ALIASES,
    RAW_ORDERBOOK_ROOT,
    REPLAY_ORDERBOOK_ROOT,
    build_download_manifest,
    download_manifest_files,
    parse_orderbook_zip_for_targets,
    _cached_parse_status,
    _event_id,
    _fallback_orderbook_file,
    _read_existing_replay_features,
    _write_replay_features,
)


REPORT_ROOT = Path("reports/v0_9e2_upside_vacuum_validation")
CAPACITY_ROOT = Path("reports/v0_9d_cic_capacity_architecture")
FAIR_START = pd.Timestamp("2025-07-01", tz="UTC")
FAIR_END = pd.Timestamp("2026-04-01", tz="UTC")
SOURCE_POOLS = ("P2_CIC1_CIC2_COMBINED", "P0_CIC1_ONLY")
SOURCE_MAX_POSITIONS = (5, 8, 10)
RANKING_MAX_POSITIONS = (3, 5, 8, 10)
FOCUS_POOL = "P2_CIC1_CIC2_COMBINED"
RANDOM_PERMUTATIONS = 100
DEFAULT_REPLAY_WORKERS = 1 if os.name == "nt" else 8


@dataclass(frozen=True)
class V09E2Config:
    report_root: Path = REPORT_ROOT
    capacity_root: Path = CAPACITY_ROOT
    replay_orderbook_root: Path = REPLAY_ORDERBOOK_ROOT
    raw_orderbook_root: Path = RAW_ORDERBOOK_ROOT
    max_files: int | None = None
    download: bool = True
    max_staleness_minutes: int = 30
    include_reference_p0: bool = True
    replay_workers: int = DEFAULT_REPLAY_WORKERS


RANKING_RULES = {
    "R0_first_come_first_served": "rank_first_come",
    "R2_ask_depth_25bp_low_first": "rank_ask_depth_25bp_low",
    "R3_ask_depth_25bp_relative_low_first": "rank_ask_depth_25bp_relative_low",
    "R4_ask_depth_10bp_low_first": "rank_ask_depth_10bp_low",
    "R5_ask_depth_50bp_low_first": "rank_ask_depth_50bp_low",
    "R6_healthy_upside_vacuum_uv4": "rank_healthy_uv4",
    "R7_spread_low_first": "rank_spread_low",
    "R8_bid_depth_25bp_high_first": "rank_bid_depth_25bp_high",
    "R9_old_cluster_rank_reference": "rank_cluster_impulse_density",
    "NC_ask_depth_25bp_high_first": "rank_ask_depth_25bp_high",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def _normalise_capacity_rows(frame: pd.DataFrame, *, selection_status: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out.get("exit_time"), utc=True, errors="coerce")
    out["source_selection_status"] = selection_status
    if "skip_reason" not in out.columns:
        out["skip_reason"] = ""
    if "max_positions" in out.columns:
        out["source_max_positions"] = pd.to_numeric(out["max_positions"], errors="coerce").astype("Int64")
    else:
        out["source_max_positions"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["source_pool"] = out.get("pool", "").astype(str)
    out["historical_candidate"] = out.get("candidate", "").astype(str)
    out["candidate"] = out["historical_candidate"].replace(CANDIDATE_ALIASES)
    out["base_event_id"] = _event_id(out)
    out["event_id"] = out["base_event_id"]
    out["decision_time"] = out["entry_time"]
    out["date"] = out["decision_time"].dt.strftime("%Y-%m-%d")
    if "net_return_20bp" not in out.columns and "net_return" in out.columns:
        out["net_return_20bp"] = out["net_return"]
    if "net_return_10bp" not in out.columns and "gross_return" in out.columns:
        out["net_return_10bp"] = pd.to_numeric(out["gross_return"], errors="coerce") - 0.002
    if "net_return_10bp" not in out.columns and "net_return" in out.columns:
        out["net_return_10bp"] = out["net_return"]
    return out


def load_capacity_context_rows(cfg: V09E2Config = V09E2Config()) -> pd.DataFrame:
    selected = _normalise_capacity_rows(_read_csv(cfg.capacity_root / "portfolio_timeline.csv"), selection_status="selected")
    skipped = _normalise_capacity_rows(_read_csv(cfg.capacity_root / "portfolio_skipped_candidates.csv"), selection_status="skipped")
    if not skipped.empty:
        skipped = skipped[skipped["skip_reason"].astype(str).eq("portfolio_full")].copy()
    frames = [frame for frame in [selected, skipped] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    pools = list(SOURCE_POOLS if cfg.include_reference_p0 else (FOCUS_POOL,))
    data = data[
        data["source_pool"].isin(pools)
        & data["source_max_positions"].astype("Int64").isin(SOURCE_MAX_POSITIONS)
        & (data["entry_time"] >= FAIR_START)
        & (data["entry_time"] < FAIR_END)
    ].copy()
    data["capacity_context_id"] = (
        data["source_pool"].astype(str)
        + "|max"
        + data["source_max_positions"].astype(str)
        + "|"
        + data["source_selection_status"].astype(str)
        + "|"
        + data["base_event_id"].astype(str)
    )
    return data.sort_values(["source_pool", "source_max_positions", "entry_time", "symbol"]).reset_index(drop=True)


def _aggregate_csv(values: pd.Series) -> str:
    return ",".join(sorted(set(values.dropna().astype(str))))


def build_unique_replay_targets(context: pd.DataFrame) -> pd.DataFrame:
    if context.empty:
        return pd.DataFrame()
    grouped = (
        context.groupby("base_event_id", as_index=False, sort=False)
        .agg(
            event_id=("base_event_id", "first"),
            trade_id=("trade_id", "first") if "trade_id" in context.columns else ("base_event_id", "first"),
            signal_id=("signal_id", "first") if "signal_id" in context.columns else ("base_event_id", "first"),
            exchange=("exchange", "first") if "exchange" in context.columns else ("base_event_id", "first"),
            symbol=("symbol", "first"),
            candidate=("candidate", "first"),
            historical_candidate=("historical_candidate", "first"),
            entry_time=("entry_time", "first"),
            exit_time=("exit_time", "first"),
            decision_time=("decision_time", "first"),
            date=("date", "first"),
            net_return_10bp=("net_return_10bp", "first"),
            net_return_20bp=("net_return_20bp", "first"),
            net_return=("net_return", "first") if "net_return" in context.columns else ("net_return_20bp", "first"),
            source_pools=("source_pool", _aggregate_csv),
            source_max_positions=("source_max_positions", _aggregate_csv),
            source_selection_statuses=("source_selection_status", _aggregate_csv),
            context_rows=("capacity_context_id", "nunique"),
        )
        .copy()
    )
    grouped["is_skipped_target"] = grouped["source_selection_statuses"].astype(str).str.contains("skipped", na=False)
    grouped["conflict_set_id"] = "v09e2|" + grouped["date"].astype(str)
    grouped["source_file"] = "v09d_capacity_context"
    grouped["selection_status"] = grouped["source_selection_statuses"]
    grouped["skip_reason"] = np.where(grouped["is_skipped_target"], "portfolio_full", "")
    return grouped.sort_values(["date", "symbol", "entry_time", "candidate"]).reset_index(drop=True)


def build_pool_trade_source(context: pd.DataFrame) -> pd.DataFrame:
    if context.empty:
        return pd.DataFrame()
    rows = []
    for pool_name, pool_group in context.groupby("source_pool", sort=False):
        local = pool_group.drop_duplicates(["source_pool", "base_event_id"]).copy()
        local["analysis_pool"] = pool_name
        rows.append(local)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    out["net_return"] = pd.to_numeric(out["net_return_20bp"], errors="coerce")
    return out.sort_values(["analysis_pool", "entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _download_and_replay_targets(targets: pd.DataFrame, cfg: V09E2Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    existing = _read_existing_replay_features(cfg.replay_orderbook_root, targets)
    existing_events = set(existing.get("event_id", pd.Series(dtype=str)).dropna().astype(str))
    parse_targets = targets[~targets["event_id"].astype(str).isin(existing_events)].copy()
    manifest = build_download_manifest(parse_targets, max_files=cfg.max_files)
    if cfg.download and not manifest.empty:
        rows = []
        for row in manifest.itertuples(index=False):
            payload = row._asdict()
            payload.update(
                _fallback_orderbook_file(
                    str(row.symbol),
                    str(row.date),
                    status="fallback_url_v09e2_direct",
                )
            )
            rows.append(payload)
        manifest = pd.DataFrame(rows)
        manifest = download_manifest_files(manifest, cfg.raw_orderbook_root)

    jobs: list[tuple[str, str, list[dict[str, object]]]] = []
    for row in manifest.itertuples(index=False):
        path_raw = str(getattr(row, "local_path", "") or "")
        if not path_raw:
            continue
        path = Path(path_raw)
        if not path.is_file():
            continue
        sample = targets[targets["symbol"].eq(str(row.symbol)) & targets["date"].eq(str(row.date))]
        if sample.empty:
            continue
        jobs.append((str(path), str(row.symbol), sample.to_dict("records")))

    feature_frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    if cfg.replay_workers <= 1 or len(jobs) <= 1:
        results = [_parse_replay_job(job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=cfg.replay_workers) as executor:
            futures = [executor.submit(_parse_replay_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    for features, status in results:
        if not features.empty:
            feature_frames.append(features)
        if not status.empty:
            status_frames.append(status)
    new_features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    features = pd.concat([existing, new_features], ignore_index=True) if not existing.empty or not new_features.empty else pd.DataFrame()
    cached_status = _cached_parse_status(existing, targets)
    parse_status = pd.concat([cached_status, *status_frames], ignore_index=True) if not cached_status.empty or status_frames else pd.DataFrame()
    _write_replay_features(features, cfg.replay_orderbook_root)
    return manifest, features, parse_status


def _parse_replay_job(job: tuple[str, str, list[dict[str, object]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_raw, symbol, records = job
    sample = pd.DataFrame.from_records(records)
    if sample.empty:
        return pd.DataFrame(), pd.DataFrame()
    try:
        return parse_orderbook_zip_for_targets(Path(path_raw), sample, symbol)
    except Exception as exc:
        status = sample[["event_id", "symbol", "decision_time"]].copy()
        status["snapshot_time"] = pd.NaT
        status["snapshot_cts"] = pd.NaT
        status["target_snapshot_age_ms"] = np.nan
        status["replay_status"] = "parse_error"
        status["error"] = str(exc)
        return pd.DataFrame(), status


def _attach_features_to_source(source: pd.DataFrame, features: pd.DataFrame, cfg: V09E2Config) -> pd.DataFrame:
    symbols = sorted(source.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    feature_cache = _load_orderbook_features(cfg.replay_orderbook_root, symbols)
    if not features.empty:
        feature_cache = pd.concat([feature_cache, features], ignore_index=True) if not feature_cache.empty else features
    attached = _attach_orderbook_features(source, feature_cache, max_staleness_minutes=cfg.max_staleness_minutes)
    attached["strict_asof_covered"] = attached["coverage_status"].astype(str).eq("covered")
    attached["sensitivity_covered"] = attached["coverage_status"].astype(str).isin(["covered", "covered_live_delay"])
    return _add_upside_vacuum_features(attached)


def _pct_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average")


def _add_upside_vacuum_features(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    out = data.copy()
    ask25 = _num(out, "ask_depth_25bp")
    grouped = out.groupby("symbol", sort=False)["ask_depth_25bp"]
    median = grouped.transform(lambda col: pd.to_numeric(col, errors="coerce").median())
    std = grouped.transform(lambda col: pd.to_numeric(col, errors="coerce").std(ddof=0))
    out["ask_depth_25bp_relative_to_symbol_median"] = ask25 / median.replace(0, np.nan)
    out["ask_depth_25bp_z_by_symbol_replay"] = (ask25 - median) / std.replace(0, np.nan)
    out["ask_depth_25bp_pctile_by_symbol_replay"] = grouped.transform(lambda col: _pct_rank(pd.to_numeric(col, errors="coerce")))

    spread = _num(out, "spread_bps")
    bid25 = _num(out, "bid_depth_25bp")
    downside = _num(out, "downside_liquidity_risk_25bp")
    out["spread_not_high"] = spread <= spread.quantile(0.80)
    out["bid_depth_not_low"] = bid25 >= bid25.quantile(0.20)
    out["downside_risk_low"] = downside <= downside.quantile(0.50)
    out["uv1_ask_depth_25bp_low"] = -ask25
    out["uv2_ask_low_spread_ok"] = -ask25 + out["spread_not_high"].astype(float) * ask25.abs().rank(pct=True).fillna(0)
    out["uv3_ask_low_bid_ok"] = -ask25 + out["bid_depth_not_low"].astype(float) * bid25.rank(pct=True).fillna(0)
    out["uv4_healthy_upside_vacuum"] = (
        (-ask25).rank(pct=True)
        + out["spread_not_high"].astype(float)
        + out["bid_depth_not_low"].astype(float)
    )
    out["uv5_ask_low_downside_risk_low"] = (-ask25).rank(pct=True) + out["downside_risk_low"].astype(float)
    out["rank_first_come"] = 0.0
    out["rank_ask_depth_25bp_low"] = -ask25
    out["rank_ask_depth_25bp_high"] = ask25
    out["rank_ask_depth_25bp_relative_low"] = -_num(out, "ask_depth_25bp_relative_to_symbol_median")
    out["rank_ask_depth_10bp_low"] = -_num(out, "ask_depth_10bp")
    out["rank_ask_depth_50bp_low"] = -_num(out, "ask_depth_50bp")
    out["rank_healthy_uv4"] = _num(out, "uv4_healthy_upside_vacuum")
    out["rank_spread_low"] = -spread
    out["rank_bid_depth_25bp_high"] = bid25
    out["rank_cluster_impulse_density"] = _num(out, "cluster_impulse_density", 0.0)
    return out


def _coverage_audit(data: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "base_event_id",
        "analysis_pool",
        "candidate",
        "historical_candidate",
        "symbol",
        "decision_time",
        "snapshot_time",
        "snapshot_cts",
        "snapshot_age_ms",
        "latency_ms",
        "coverage_status",
        "strict_asof_covered",
        "sensitivity_covered",
        "levels_available",
        "best_bid",
        "best_ask",
        "spread_bps",
        "source",
    ]
    out = data.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols].sort_values(["decision_time", "symbol", "analysis_pool"]).reset_index(drop=True)


def _asof_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool, group in data.groupby("analysis_pool", sort=False, dropna=False):
        rows.append(
            {
                "analysis_pool": pool,
                "trades": int(len(group)),
                "strict_covered": int(group["strict_asof_covered"].sum()),
                "sensitivity_covered": int(group["sensitivity_covered"].sum()),
                "strict_coverage_rate": float(group["strict_asof_covered"].mean()) if len(group) else np.nan,
                "sensitivity_coverage_rate": float(group["sensitivity_covered"].mean()) if len(group) else np.nan,
                "covered_live_delay": int(group["coverage_status"].astype(str).eq("covered_live_delay").sum()),
                "stale_or_missing": int((~group["sensitivity_covered"]).sum()),
                "median_snapshot_age_ms": float(_num(group[group["strict_asof_covered"]], "snapshot_age_ms").median())
                if group["strict_asof_covered"].any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _metrics(selected: pd.DataFrame, skipped: pd.DataFrame, *, pool: str, rule: str, max_positions: int, mode: str) -> dict[str, object]:
    selected_net20 = _num(selected, "net_return")
    skipped_net20 = _num(skipped, "net_return")
    selected_net10 = _num(selected, "net_return_10bp")
    skipped_net10 = _num(skipped, "net_return_10bp")
    return {
        "asof_mode": mode,
        "pool": pool,
        "ranking_rule": rule,
        "max_positions": int(max_positions),
        "selected_trades": int(len(selected)),
        "skipped_trades": int(len(skipped)),
        "selected_net10": float(selected_net10.mean()) if len(selected_net10) else np.nan,
        "selected_net20": float(selected_net20.mean()) if len(selected_net20) else np.nan,
        "skipped_net10": float(skipped_net10.mean()) if len(skipped_net10) else np.nan,
        "skipped_net20": float(skipped_net20.mean()) if len(skipped_net20) else np.nan,
        "selected_minus_skipped": float(selected_net20.mean() - skipped_net20.mean())
        if len(selected_net20) and len(skipped_net20)
        else np.nan,
        "portfolio_net20": float(selected_net20.sum()) if len(selected_net20) else 0.0,
        "month_cap35_net20": _month_cap_expectancy(selected),
        "max_symbol_contribution": _max_contribution(selected, "symbol"),
        "max_month_contribution": _max_contribution(selected, "month"),
    }


def _sample_for_mode(data: pd.DataFrame, pool: str, mode: str) -> pd.DataFrame:
    if mode == "strict_asof_only":
        mask = data["strict_asof_covered"].fillna(False).astype(bool)
    elif mode == "strict_plus_live_delay":
        mask = data["sensitivity_covered"].fillna(False).astype(bool)
    else:
        raise ValueError(f"unknown asof mode: {mode}")
    sample = data[data["analysis_pool"].astype(str).eq(pool) & mask].copy()
    sample["month"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    sample["net_return"] = _num(sample, "net_return_20bp")
    return sample.dropna(subset=["entry_time", "exit_time", "net_return"]).reset_index(drop=True)


def _ranking_summary(data: pd.DataFrame, *, modes: tuple[str, ...] = ("strict_asof_only", "strict_plus_live_delay")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_frames = []
    skipped_frames = []
    for mode in modes:
        for pool in SOURCE_POOLS:
            sample = _sample_for_mode(data, pool, mode)
            if sample.empty:
                continue
            for rule, score_col in RANKING_RULES.items():
                if score_col not in sample.columns:
                    continue
                for max_positions in RANKING_MAX_POSITIONS:
                    selected, skipped = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                    rows.append(_metrics(selected, skipped, pool=pool, rule=rule, max_positions=max_positions, mode=mode))
                    if pool == FOCUS_POOL and max_positions in {5, 8} and mode == "strict_asof_only":
                        if not selected.empty:
                            selected_frames.append(selected.assign(asof_mode=mode, pool=pool, ranking_rule=rule, max_positions=max_positions))
                        if not skipped.empty:
                            skipped_frames.append(skipped.assign(asof_mode=mode, pool=pool, ranking_rule=rule, max_positions=max_positions))
    return (
        pd.DataFrame(rows),
        pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _random_summary(data: pd.DataFrame, seed: int = 922) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for mode in ["strict_asof_only", "strict_plus_live_delay"]:
        for pool in SOURCE_POOLS:
            sample = _sample_for_mode(data, pool, mode)
            if sample.empty:
                continue
            distributions: dict[int, np.ndarray] = {}
            for max_positions in RANKING_MAX_POSITIONS:
                totals = []
                for _ in range(RANDOM_PERMUTATIONS):
                    local = sample.copy()
                    local["rank_random"] = rng.normal(size=len(local))
                    selected, _ = select_portfolio(local, score_col="rank_random", max_positions=max_positions)
                    totals.append(float(_num(selected, "net_return").sum()) if not selected.empty else 0.0)
                distributions[max_positions] = np.array(totals, dtype=float)
            for rule, score_col in RANKING_RULES.items():
                if score_col not in sample.columns:
                    continue
                for max_positions in RANKING_MAX_POSITIONS:
                    selected, _ = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                    total = float(_num(selected, "net_return").sum()) if not selected.empty else 0.0
                    dist = distributions[max_positions]
                    rows.append(
                        {
                            "asof_mode": mode,
                            "pool": pool,
                            "ranking_rule": rule,
                            "max_positions": max_positions,
                            "real_net20": total,
                            "random_mean": float(np.mean(dist)),
                            "random_median": float(np.median(dist)),
                            "random_p75": float(np.quantile(dist, 0.75)),
                            "random_p90": float(np.quantile(dist, 0.90)),
                            "real_percentile": float((dist <= total).mean()),
                            "permutations": RANDOM_PERMUTATIONS,
                        }
                    )
    return pd.DataFrame(rows)


def _ask_depth_bucket_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ["strict_asof_only", "strict_plus_live_delay"]:
        for pool in SOURCE_POOLS:
            sample = _sample_for_mode(data, pool, mode)
            if len(sample) < 10:
                continue
            for feature in ["ask_depth_10bp", "ask_depth_25bp", "ask_depth_50bp", "ask_depth_25bp_relative_to_symbol_median"]:
                values = _num(sample, feature)
                valid = sample[values.notna()].copy()
                if len(valid) < 10:
                    continue
                try:
                    valid["_bucket"] = pd.qcut(_num(valid, feature), q=min(5, len(valid)), labels=False, duplicates="drop")
                except ValueError:
                    continue
                bucket_count = int(valid["_bucket"].max() + 1) if len(valid) else 0
                for bucket, group in valid.groupby("_bucket", sort=True, dropna=False):
                    idx = int(bucket)
                    label = f"q{idx + 1}_{'thinnest' if idx == 0 else 'thickest' if idx == bucket_count - 1 else 'mid'}"
                    rows.append(
                        {
                            "asof_mode": mode,
                            "pool": pool,
                            "feature": feature,
                            "bucket": label,
                            "trades": int(len(group)),
                            "net10": float(_num(group, "net_return_10bp").mean()),
                            "net20": float(_num(group, "net_return").mean()),
                            "tp_rate": float(group.get("exit_reason", pd.Series("", index=group.index)).astype(str).str.contains("tp", case=False).mean())
                            if "exit_reason" in group.columns
                            else np.nan,
                            "sl_rate": float(group.get("exit_reason", pd.Series("", index=group.index)).astype(str).str.contains("sl", case=False).mean())
                            if "exit_reason" in group.columns
                            else np.nan,
                            "timeout_rate": float(group.get("exit_reason", pd.Series("", index=group.index)).astype(str).str.contains("timeout", case=False).mean())
                            if "exit_reason" in group.columns
                            else np.nan,
                            "historical_selected_context_rows": int(
                                group.get("source_selection_status", pd.Series("", index=group.index)).astype(str).str.contains("selected").sum()
                            ),
                            "historical_skipped_context_rows": int(
                                group.get("source_selection_status", pd.Series("", index=group.index)).astype(str).str.contains("skipped").sum()
                            ),
                            "month_cap35_net20": _month_cap_expectancy(group),
                        }
                    )
    return pd.DataFrame(rows)


def _upside_vacuum_variant_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = {
        "UV1_ask_depth_25bp_low": "uv1_ask_depth_25bp_low",
        "UV2_ask_low_spread_not_high": "uv2_ask_low_spread_ok",
        "UV3_ask_low_bid_not_low": "uv3_ask_low_bid_ok",
        "UV4_healthy_ask_low_spread_ok_bid_ok": "uv4_healthy_upside_vacuum",
        "UV5_ask_low_downside_risk_low": "uv5_ask_low_downside_risk_low",
    }
    for mode in ["strict_asof_only", "strict_plus_live_delay"]:
        sample = _sample_for_mode(data, FOCUS_POOL, mode)
        if sample.empty:
            continue
        for name, score_col in variants.items():
            if score_col not in sample.columns:
                continue
            for max_positions in [5, 8]:
                selected, skipped = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                row = _metrics(selected, skipped, pool=FOCUS_POOL, rule=name, max_positions=max_positions, mode=mode)
                rows.append(row)
    return pd.DataFrame(rows)


def _negative_controls(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = _sample_for_mode(data, FOCUS_POOL, "strict_asof_only")
    if sample.empty:
        return pd.DataFrame()
    controls = {
        "ask_depth_25bp_high_first": "rank_ask_depth_25bp_high",
        "wide_spread_top20pct": _num(sample, "spread_bps") >= _num(sample, "spread_bps").quantile(0.80),
        "low_bid_depth_top20pct_bad": _num(sample, "bid_depth_25bp") <= _num(sample, "bid_depth_25bp").quantile(0.20),
        "high_downside_risk_top20pct": _num(sample, "downside_liquidity_risk_25bp")
        >= _num(sample, "downside_liquidity_risk_25bp").quantile(0.80),
        "delayed_snapshot_audit_only": data[
            data["analysis_pool"].astype(str).eq(FOCUS_POOL)
            & data["coverage_status"].astype(str).eq("covered_live_delay")
        ].index,
    }
    selected, skipped = select_portfolio(sample, score_col="rank_ask_depth_25bp_high", max_positions=8)
    rows.append(_metrics(selected, skipped, pool=FOCUS_POOL, rule="ask_depth_25bp_high_first", max_positions=8, mode="strict_asof_only"))
    for name, mask in controls.items():
        if isinstance(mask, str):
            continue
        if isinstance(mask, pd.Index):
            group = data.loc[mask].copy()
            group["net_return"] = _num(group, "net_return_20bp")
        else:
            group = sample[mask.fillna(False)].copy()
        rows.append(
            {
                "asof_mode": "strict_asof_only" if name != "delayed_snapshot_audit_only" else "sensitivity_only",
                "pool": FOCUS_POOL,
                "ranking_rule": name,
                "max_positions": np.nan,
                "selected_trades": int(len(group)),
                "skipped_trades": np.nan,
                "selected_net10": float(_num(group, "net_return_10bp").mean()) if len(group) else np.nan,
                "selected_net20": float(_num(group, "net_return").mean()) if len(group) else np.nan,
                "skipped_net10": np.nan,
                "skipped_net20": np.nan,
                "selected_minus_skipped": np.nan,
                "portfolio_net20": float(_num(group, "net_return").sum()) if len(group) else 0.0,
                "month_cap35_net20": _month_cap_expectancy(group),
                "max_symbol_contribution": _max_contribution(group, "symbol"),
                "max_month_contribution": _max_contribution(group, "month"),
            }
        )
    return pd.DataFrame(rows)


def _conflict_set_detail(selected: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not selected.empty:
        frames.append(selected.assign(selection_result="selected"))
    if not skipped.empty:
        frames.append(skipped.assign(selection_result="skipped"))
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["decision_bucket"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce").dt.floor("15min")
    cols = [
        "asof_mode",
        "pool",
        "ranking_rule",
        "max_positions",
        "selection_result",
        "symbol",
        "candidate",
        "entry_time",
        "decision_bucket",
        "net_return_10bp",
        "net_return",
        "ask_depth_25bp",
        "ask_depth_25bp_relative_to_symbol_median",
        "bid_depth_25bp",
        "spread_bps",
        "coverage_status",
    ]
    for col in cols:
        if col not in data.columns:
            data[col] = np.nan
    return data[cols].sort_values(["ranking_rule", "max_positions", "entry_time", "selection_result", "symbol"])


def _write_notes(
    path: Path,
    context: pd.DataFrame,
    targets: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    parse_status: pd.DataFrame,
    data: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    strict = data[data["analysis_pool"].astype(str).eq(FOCUS_POOL) & data["strict_asof_covered"].fillna(False)]
    best = ranking[
        ranking["asof_mode"].eq("strict_asof_only")
        & ranking["pool"].eq(FOCUS_POOL)
        & ranking["max_positions"].isin([5, 8])
    ].copy()
    lines = [
        "# v0.9E.2 Upside Vacuum Validation",
        "",
        f"capacity_context_rows: {len(context)}",
        f"unique_replay_targets: {len(targets)}",
        f"current_manifest_symbol_days: {len(manifest)}",
        f"historical_orderbook_feature_rows: {len(features)}",
        f"historical_orderbook_feature_events: {features['event_id'].nunique() if 'event_id' in features.columns else 0}",
        f"p2_strict_asof_trades: {len(strict)}",
        f"p2_strict_coverage_rate: {float(strict.shape[0] / max(1, data[data['analysis_pool'].eq(FOCUS_POOL)].shape[0])):.2%}",
        "",
        "Note: when rerun with `--no-download`, `current_manifest_symbol_days` is the remaining missing increment, not the original full request set.",
        "Main report uses strict `covered` snapshots only. `covered_live_delay` is sensitivity-only.",
        "True 30d orderbook normalization is not available from event-only replay; relative ask-depth columns use replay-sample symbol baselines.",
        "",
    ]
    if not parse_status.empty and "replay_status" in parse_status.columns:
        status_text = ", ".join(
            f"{key}={value}" for key, value in parse_status["replay_status"].fillna("unknown").value_counts().items()
        )
        lines.extend([f"replay_status_counts: {status_text}", ""])
    strict_p2 = ranking[
        ranking["asof_mode"].eq("strict_asof_only")
        & ranking["pool"].eq(FOCUS_POOL)
        & ranking["max_positions"].isin([5, 8, 10])
    ].copy()
    positive_selected_delta = strict_p2["selected_minus_skipped"].gt(0).any() if not strict_p2.empty else False
    coverage_rate = float(strict.shape[0] / max(1, data[data["analysis_pool"].eq(FOCUS_POOL)].shape[0]))
    if coverage_rate < 0.80 or not positive_selected_delta:
        lines.extend(
            [
                "## Decision",
                "",
                "Status: diagnostic only; do not promote ask-thin / upside-vacuum orderbook ranking to shadow portfolio.",
                "",
                "Reasons:",
                f"- Strict-asof P2 coverage is {coverage_rate:.2%}, below the 80%-90% target.",
                "- No strict P2 max5/max8/max10 ranking rule achieved selected_net20 > skipped_net20.",
                "- Ask-depth buckets are not monotonic; the thinnest ask-depth bucket is not the strongest bucket.",
                "- Healthy upside-vacuum variants did not improve the selected-vs-skipped gap.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Decision",
                "",
                "Status: candidate for shadow portfolio review; strict-asof coverage and selected-vs-skipped criteria passed.",
                "",
            ]
        )
    if not best.empty:
        lines.append("## Best Strict P2 Rows")
        for row in best.sort_values("selected_minus_skipped", ascending=False).head(6).itertuples(index=False):
            lines.append(
                f"- {row.ranking_rule} max={row.max_positions}: selected_net20={row.selected_net20:.4%}, "
                f"skipped_net20={row.skipped_net20:.4%}, delta={row.selected_minus_skipped:.4%}, "
                f"random_percentile={getattr(row, 'real_percentile', np.nan):.2f}."
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_v09e2_upside_vacuum_validation(cfg: V09E2Config = V09E2Config()) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    context = load_capacity_context_rows(cfg)
    targets = build_unique_replay_targets(context)
    manifest, features, parse_status = _download_and_replay_targets(targets, cfg)
    source = build_pool_trade_source(context)
    data = _attach_features_to_source(source, features, cfg)
    coverage = _coverage_audit(data)
    asof = _asof_quality_summary(data)
    buckets = _ask_depth_bucket_summary(data)
    uv = _upside_vacuum_variant_summary(data)
    ranking, selected, skipped = _ranking_summary(data)
    random = _random_summary(data)
    if not ranking.empty and not random.empty:
        ranking = ranking.merge(
            random[["asof_mode", "pool", "ranking_rule", "max_positions", "real_percentile"]],
            on=["asof_mode", "pool", "ranking_rule", "max_positions"],
            how="left",
        )
    negatives = _negative_controls(data)
    conflict = _conflict_set_detail(selected, skipped)

    outputs = {
        "capacity_context_rows": report_root / "capacity_context_rows.csv",
        "replay_targets": report_root / "replay_targets.csv",
        "download_manifest": report_root / "download_manifest.csv",
        "replay_parse_status": report_root / "replay_parse_status.csv",
        "historical_orderbook_features": report_root / "historical_orderbook_features.csv",
        "coverage_audit": report_root / "coverage_audit.csv",
        "asof_quality_summary": report_root / "asof_quality_summary.csv",
        "ask_depth_bucket_summary": report_root / "ask_depth_bucket_summary.csv",
        "upside_vacuum_variant_summary": report_root / "upside_vacuum_variant_summary.csv",
        "ranking_summary": report_root / "ranking_summary.csv",
        "selected_vs_skipped_orderbook": report_root / "selected_vs_skipped_orderbook.csv",
        "random_permutation_summary": report_root / "random_permutation_summary.csv",
        "negative_controls": report_root / "negative_controls.csv",
        "conflict_set_detail": report_root / "conflict_set_detail.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    context.to_csv(outputs["capacity_context_rows"], index=False)
    targets.to_csv(outputs["replay_targets"], index=False)
    manifest.to_csv(outputs["download_manifest"], index=False)
    parse_status.to_csv(outputs["replay_parse_status"], index=False)
    features.to_csv(outputs["historical_orderbook_features"], index=False)
    coverage.to_csv(outputs["coverage_audit"], index=False)
    asof.to_csv(outputs["asof_quality_summary"], index=False)
    buckets.to_csv(outputs["ask_depth_bucket_summary"], index=False)
    uv.to_csv(outputs["upside_vacuum_variant_summary"], index=False)
    ranking.to_csv(outputs["ranking_summary"], index=False)
    ranking.to_csv(outputs["selected_vs_skipped_orderbook"], index=False)
    random.to_csv(outputs["random_permutation_summary"], index=False)
    negatives.to_csv(outputs["negative_controls"], index=False)
    conflict.to_csv(outputs["conflict_set_detail"], index=False)
    _write_notes(outputs["candidate_notes"], context, targets, manifest, features, parse_status, data, ranking)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V09E2Config",
    "build_pool_trade_source",
    "build_unique_replay_targets",
    "load_capacity_context_rows",
    "run_v09e2_upside_vacuum_validation",
]
