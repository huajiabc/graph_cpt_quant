from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v09b import select_portfolio
from pressure_graph.reports.v09c import POOLS, MAX_POSITIONS, FOCUS_COST


REPORT_ROOT = Path("reports/v0_9e_orderbook_capacity_ranking")
SOURCE_ROOT = Path("reports/v0_8_orderflow_shadow")
ORDERBOOK_ROOT = Path("data/orderbook/v0_8_5/bybit")
MAX_STALENESS_MINUTES = 30
LIVE_DELAY_TOLERANCE_MS = 5_000
MIN_LEVELS_AVAILABLE = 20
MAX_SPREAD_BPS = 100.0
MIN_DIAGNOSTIC_TRADES = 50
MIN_DECISION_TRADES = 100

ORDERBOOK_FEATURES = [
    "best_bid",
    "best_ask",
    "mid",
    "spread_bps",
    "bid_levels_available",
    "ask_levels_available",
    "levels_available",
    "bid_depth_5bp",
    "ask_depth_5bp",
    "imbalance_5bp",
    "bid_depth_10bp",
    "ask_depth_10bp",
    "imbalance_10bp",
    "bid_depth_20bp",
    "ask_depth_20bp",
    "imbalance_20bp",
    "bid_depth_25bp",
    "ask_depth_25bp",
    "imbalance_25bp",
    "top5_bid_notional",
    "top5_ask_notional",
    "top5_imbalance",
    "ask_wall_20bp_ratio",
    "bid_wall_20bp_ratio",
    "buy_impact_10k",
    "buy_impact_50k",
    "buy_impact_100k",
    "sell_impact_10k",
    "sell_impact_50k",
    "sell_impact_100k",
    "upside_vacuum_25bp",
    "downside_liquidity_risk_25bp",
    "entry_book_quality_score",
    "reclaim_spread_bps",
    "reclaim_bid_depth_25bp",
    "reclaim_ask_depth_25bp",
    "reclaim_imbalance_25bp",
    "reclaim_upside_vacuum",
    "reclaim_downside_support",
]

RANKING_RULES = {
    "R0_cluster_rank_reference": "rank_cluster_impulse_density",
    "R1_spread_low": "rank_spread_low",
    "R2_bid_ask_imbalance_25bp_high": "rank_imbalance_25bp",
    "R3_ask_depth_25bp_low": "rank_ask_depth_25bp_low",
    "R4_bid_depth_25bp_high": "rank_bid_depth_25bp",
    "R5_downside_liquidity_risk_low": "rank_downside_liquidity_risk_low",
    "R6_upside_vacuum_high": "rank_upside_vacuum",
    "R7_book_quality_composite": "rank_book_quality_composite",
    "R8_orderflow_orderbook_composite": "rank_orderflow_orderbook_composite",
}

COVERAGE_AUDIT_COLUMNS = [
    "event_id",
    "trade_id",
    "signal_id",
    "candidate",
    "pool_hint",
    "symbol",
    "decision_time",
    "snapshot_time",
    "snapshot_cts",
    "latency_ms",
    "snapshot_age_ms",
    "receive_delay_ms",
    "coverage_status",
    "levels_available",
    "bid_levels_available",
    "ask_levels_available",
    "best_bid",
    "best_ask",
    "spread_bps",
    "source",
]

RANKING_COLUMNS = [
    "ranking_rule",
    "pool",
    "max_positions",
    "selected_trades",
    "skipped_trades",
    "selected_net10",
    "selected_net20",
    "skipped_net10",
    "skipped_net20",
    "selected_minus_skipped",
    "portfolio_net20",
    "month_cap35_net20",
    "max_symbol_contribution",
    "orderbook_coverage_rate",
    "avg_orderbook_staleness_minutes",
    "random_percentile",
]


def _read_trades(source_root: Path) -> pd.DataFrame:
    path = source_root / "orderflow_shadow_trades.csv"
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _feature_cache_path(root: Path, symbol: str) -> Path:
    return root / "features" / f"{symbol}.parquet"


def _level_cache_path(root: Path, symbol: str) -> Path:
    return root / "snapshots" / f"{symbol}.parquet"


def _load_level_counts(orderbook_root: Path, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        path = _level_cache_path(orderbook_root, symbol)
        if not path.exists():
            continue
        frame = read_parquet(path)
        if frame.empty or "snapshot_time" not in frame.columns:
            continue
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].astype(str)
        frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], utc=True, errors="coerce")
        grouped = (
            frame.dropna(subset=["symbol", "snapshot_time"])
            .pivot_table(
                index=["symbol", "snapshot_time"],
                columns="side",
                values="level",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        grouped["bid_levels_available_from_levels"] = pd.to_numeric(grouped.get("bid", 0), errors="coerce")
        grouped["ask_levels_available_from_levels"] = pd.to_numeric(grouped.get("ask", 0), errors="coerce")
        grouped["levels_available_from_levels"] = (
            grouped["bid_levels_available_from_levels"] + grouped["ask_levels_available_from_levels"]
        )
        frames.append(
            grouped[
                [
                    "symbol",
                    "snapshot_time",
                    "bid_levels_available_from_levels",
                    "ask_levels_available_from_levels",
                    "levels_available_from_levels",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_orderbook_features(orderbook_root: Path, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        path = _feature_cache_path(orderbook_root, symbol)
        if not path.exists():
            continue
        frame = read_parquet(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].astype(str)
        frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], utc=True, errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["symbol", "snapshot_time"]).sort_values(["symbol", "snapshot_time"])
    if "exchange_ts" in out.columns:
        out["snapshot_cts"] = pd.to_datetime(out["exchange_ts"], utc=True, errors="coerce")
    else:
        out["snapshot_cts"] = pd.NaT
    level_counts = _load_level_counts(orderbook_root, symbols)
    if not level_counts.empty:
        out = out.merge(level_counts, on=["symbol", "snapshot_time"], how="left")
        for base in ["bid_levels_available", "ask_levels_available", "levels_available"]:
            fallback = f"{base}_from_levels"
            if base not in out.columns:
                out[base] = out[fallback]
            else:
                out[base] = pd.to_numeric(out[base], errors="coerce").fillna(
                    pd.to_numeric(out[fallback], errors="coerce")
                )
            out = out.drop(columns=[fallback], errors="ignore")
    out["source"] = "bybit_rest_orderbook_snapshot"
    return out


def _prepare_trades(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    out = data.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    out["symbol"] = out["symbol"].astype(str)
    out["candidate"] = out["candidate"].astype(str)
    out["decision_time"] = out["entry_time"]
    if "trade_id" not in out.columns:
        out["trade_id"] = ""
    if "signal_id" not in out.columns:
        out["signal_id"] = ""
    out["event_id"] = np.where(
        out["trade_id"].astype(str).str.len() > 0,
        out["trade_id"].astype(str),
        out["symbol"].astype(str) + "|" + out["candidate"].astype(str) + "|" + out["entry_time"].astype(str),
    )
    out["net_return"] = _num(out, f"net_return_{FOCUS_COST}bp")
    if "holding_minutes" not in out.columns:
        out["holding_minutes"] = (out["exit_time"] - out["entry_time"]).dt.total_seconds() / 60.0
    return out.dropna(subset=["entry_time", "exit_time", "net_return"]).reset_index(drop=True)


def _add_orderbook_alias_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    out = features.copy()
    for col in ORDERBOOK_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    out["reclaim_spread_bps"] = pd.to_numeric(out.get("spread_bps"), errors="coerce")
    out["reclaim_bid_depth_25bp"] = pd.to_numeric(out.get("bid_depth_25bp"), errors="coerce")
    out["reclaim_ask_depth_25bp"] = pd.to_numeric(out.get("ask_depth_25bp"), errors="coerce")
    out["reclaim_imbalance_25bp"] = pd.to_numeric(out.get("imbalance_25bp"), errors="coerce")
    out["reclaim_upside_vacuum"] = pd.to_numeric(out.get("upside_vacuum_25bp"), errors="coerce")
    out["reclaim_downside_support"] = pd.to_numeric(out.get("bid_depth_25bp"), errors="coerce")
    return out


def _coverage_status(
    row: pd.Series,
    decision_time: pd.Timestamp,
    *,
    live_delay: bool,
    max_staleness_minutes: int,
) -> tuple[str, float, float]:
    snapshot_time = pd.to_datetime(row.get("snapshot_time"), utc=True, errors="coerce")
    snapshot_cts = pd.to_datetime(row.get("snapshot_cts"), utc=True, errors="coerce")
    asof_time = snapshot_cts if pd.notna(snapshot_cts) else snapshot_time
    snapshot_age_ms = (decision_time - asof_time).total_seconds() * 1000 if pd.notna(asof_time) else np.nan
    latency_ms = (snapshot_time - snapshot_cts).total_seconds() * 1000 if pd.notna(snapshot_time) and pd.notna(snapshot_cts) else np.nan
    if pd.isna(snapshot_time):
        return "no_snapshot", snapshot_age_ms, latency_ms
    if snapshot_age_ms < -LIVE_DELAY_TOLERANCE_MS:
        return "future_snapshot_rejected", snapshot_age_ms, latency_ms
    if snapshot_age_ms > max_staleness_minutes * 60_000:
        return "stale_snapshot", snapshot_age_ms, latency_ms
    best_bid = pd.to_numeric(pd.Series([row.get("best_bid")]), errors="coerce").iloc[0]
    best_ask = pd.to_numeric(pd.Series([row.get("best_ask")]), errors="coerce").iloc[0]
    if not np.isfinite(best_bid) or not np.isfinite(best_ask) or best_bid <= 0 or best_ask <= 0:
        return "empty_book", snapshot_age_ms, latency_ms
    levels = pd.to_numeric(pd.Series([row.get("levels_available")]), errors="coerce").iloc[0]
    if np.isfinite(levels) and levels < MIN_LEVELS_AVAILABLE:
        return "levels_too_few", snapshot_age_ms, latency_ms
    spread = pd.to_numeric(pd.Series([row.get("spread_bps")]), errors="coerce").iloc[0]
    if not np.isfinite(spread) or spread < 0 or spread > MAX_SPREAD_BPS:
        return "abnormal_spread", snapshot_age_ms, latency_ms
    return ("covered_live_delay" if live_delay else "covered"), snapshot_age_ms, latency_ms


def _pick_asof_snapshot(features: pd.DataFrame, decision_time: pd.Timestamp) -> tuple[pd.Series | None, str]:
    if features.empty:
        return None, "no_snapshot"
    data = features.sort_values("snapshot_time")
    prior = data[data["snapshot_time"] <= decision_time]
    if not prior.empty:
        return prior.iloc[-1], "prior"
    if "snapshot_cts" in data.columns:
        live_delay_limit = decision_time + pd.Timedelta(milliseconds=LIVE_DELAY_TOLERANCE_MS)
        delayed = data[
            (pd.to_datetime(data["snapshot_cts"], utc=True, errors="coerce") <= decision_time)
            & (data["snapshot_time"] <= live_delay_limit)
        ]
        if not delayed.empty:
            return delayed.iloc[-1], "live_delay"
    future = data[data["snapshot_time"] > decision_time]
    if not future.empty:
        return future.iloc[0], "future_only_snapshot"
    return None, "no_snapshot"


def _attach_orderbook_features(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    *,
    max_staleness_minutes: int = MAX_STALENESS_MINUTES,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    if features.empty:
        out["orderbook_covered"] = False
        out["coverage_status"] = "no_snapshot"
        out["snapshot_time"] = pd.NaT
        out["snapshot_cts"] = pd.NaT
        out["snapshot_age_ms"] = np.nan
        out["latency_ms"] = np.nan
        out["receive_delay_ms"] = np.nan
        out["source"] = ""
        for col in ORDERBOOK_FEATURES:
            out[col] = np.nan
        out["orderbook_staleness_minutes"] = np.nan
        return out
    features = _add_orderbook_alias_features(features)
    attached_rows: list[dict[str, object]] = []
    by_symbol = {symbol: group.sort_values("snapshot_time") for symbol, group in features.groupby("symbol", sort=False)}
    for trade in out.sort_values("entry_time").itertuples(index=False):
        payload = trade._asdict()
        decision_time = pd.Timestamp(payload.get("decision_time") or payload.get("entry_time"))
        symbol_features = by_symbol.get(str(payload.get("symbol")), pd.DataFrame())
        snapshot, pick_status = _pick_asof_snapshot(symbol_features, decision_time)
        if snapshot is None:
            payload["coverage_status"] = pick_status
            payload["orderbook_covered"] = False
            payload["snapshot_time"] = pd.NaT
            payload["snapshot_cts"] = pd.NaT
            payload["snapshot_age_ms"] = np.nan
            payload["latency_ms"] = np.nan
            payload["receive_delay_ms"] = np.nan
            payload["source"] = ""
            for col in ORDERBOOK_FEATURES:
                payload[col] = np.nan
            attached_rows.append(payload)
            continue
        live_delay = pick_status == "live_delay"
        status, snapshot_age_ms, latency_ms = _coverage_status(
            snapshot,
            decision_time,
            live_delay=live_delay,
            max_staleness_minutes=max_staleness_minutes,
        )
        payload["coverage_status"] = status if pick_status != "future_only_snapshot" else "future_only_snapshot"
        payload["orderbook_covered"] = payload["coverage_status"] in {"covered", "covered_live_delay"}
        payload["snapshot_time"] = snapshot.get("snapshot_time")
        payload["snapshot_cts"] = snapshot.get("snapshot_cts")
        payload["snapshot_age_ms"] = snapshot_age_ms
        payload["latency_ms"] = latency_ms
        payload["receive_delay_ms"] = latency_ms
        payload["source"] = snapshot.get("source", "bybit_rest_orderbook_snapshot")
        for col in ORDERBOOK_FEATURES:
            payload[col] = snapshot.get(col, np.nan)
        attached_rows.append(payload)
    attached = pd.DataFrame(attached_rows)
    attached["orderbook_staleness_minutes"] = pd.to_numeric(attached.get("snapshot_age_ms"), errors="coerce") / 60_000
    return attached.sort_values("entry_time").reset_index(drop=True)


def _add_rank_scores(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    out = data.copy()
    out["rank_cluster_impulse_density"] = _num(out, "cluster_impulse_density")
    out["rank_spread_low"] = -_num(out, "spread_bps")
    out["rank_imbalance_25bp"] = _num(out, "imbalance_25bp")
    out["rank_ask_depth_25bp_low"] = -_num(out, "ask_depth_25bp")
    out["rank_bid_depth_25bp"] = _num(out, "bid_depth_25bp")
    out["rank_downside_liquidity_risk_low"] = -_num(out, "downside_liquidity_risk_25bp")
    out["rank_upside_vacuum"] = _num(out, "upside_vacuum_25bp")
    orderbook_components = [
        (-_num(out, "spread_bps")).rank(pct=True),
        _num(out, "bid_depth_25bp").rank(pct=True),
        _num(out, "imbalance_25bp").rank(pct=True),
        (-_num(out, "ask_depth_25bp")).rank(pct=True),
        (-_num(out, "downside_liquidity_risk_25bp")).rank(pct=True),
        _num(out, "upside_vacuum_25bp").rank(pct=True),
    ]
    out["rank_book_quality_composite"] = pd.concat(orderbook_components, axis=1).mean(axis=1)
    orderflow_components = [
        _num(out, "reclaim_bar_taker_buy_ratio").rank(pct=True),
        _num(out, "reclaim_bar_cvd_delta_turnover").rank(pct=True),
        _num(out, "shock_bar_taker_buy_ratio").rank(pct=True),
    ]
    out["rank_orderflow_orderbook_composite"] = pd.concat(
        [out["rank_book_quality_composite"], *orderflow_components], axis=1
    ).mean(axis=1)
    return out


def _pool(data: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    candidates = POOLS[pool_name]
    sample = data[data["candidate"].isin(candidates)].copy()
    if pool_name == "P2_CIC1_CIC2_COMBINED" and not sample.empty:
        sample["candidate_priority"] = sample["candidate"].map({"CIC1_FILTERED_MIR1": 2, "CIC2_FILTERED_MIR1": 1}).fillna(0)
        dedupe_cols = [col for col in ["signal_id", "symbol", "entry_time"] if col in sample.columns]
        if dedupe_cols:
            sample = sample.sort_values(dedupe_cols + ["candidate_priority"], ascending=[True] * len(dedupe_cols) + [False])
            sample = sample.drop_duplicates(dedupe_cols, keep="first")
    sample["pool"] = pool_name
    return sample


def _month_cap_expectancy(sample: pd.DataFrame, cap: float = 0.35) -> float:
    if sample.empty:
        return np.nan
    total = _num(sample, "net_return").sum()
    cap_value = total * cap if total > 0 else 0.0
    capped = []
    for _, group in sample.groupby("month", sort=False, dropna=False):
        value = _num(group, "net_return").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped))


def _max_contribution(sample: pd.DataFrame, group_col: str) -> float:
    if sample.empty or group_col not in sample.columns:
        return np.nan
    grouped = sample.groupby(group_col, sort=False, dropna=False)["net_return"].sum()
    total = grouped.sum()
    return float((grouped / total).abs().max()) if total else np.nan


def _metrics(selected: pd.DataFrame, skipped: pd.DataFrame, *, pool: str, rule: str, max_positions: int) -> dict[str, object]:
    selected_net = _num(selected, "net_return")
    skipped_net = _num(skipped, "net_return")
    selected_net10 = _num(selected, "net_return_10bp")
    skipped_net10 = _num(skipped, "net_return_10bp")
    return {
        "ranking_rule": rule,
        "pool": pool,
        "max_positions": max_positions,
        "selected_trades": int(len(selected)),
        "skipped_trades": int(len(skipped)),
        "selected_net10": float(selected_net10.mean()) if len(selected_net10) else np.nan,
        "selected_net20": float(selected_net.mean()) if len(selected_net) else np.nan,
        "skipped_net10": float(skipped_net10.mean()) if len(skipped_net10) else np.nan,
        "skipped_net20": float(skipped_net.mean()) if len(skipped_net) else np.nan,
        "selected_minus_skipped": float(selected_net.mean() - skipped_net.mean()) if len(selected_net) and len(skipped_net) else np.nan,
        "portfolio_net20": float(selected_net.sum()) if len(selected_net) else 0.0,
        "month_cap35_net20": _month_cap_expectancy(selected),
        "max_symbol_contribution": _max_contribution(selected, "symbol"),
        "orderbook_coverage_rate": float(selected["orderbook_covered"].mean()) if "orderbook_covered" in selected.columns and len(selected) else np.nan,
        "avg_orderbook_staleness_minutes": float(_num(selected, "orderbook_staleness_minutes").mean()) if len(selected) else np.nan,
    }


def _coverage(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        for candidate, group in sample.groupby("candidate", dropna=False):
            covered = group.get("orderbook_covered", pd.Series(False, index=group.index)).fillna(False).astype(bool)
            rows.append(
                {
                    "pool": pool_name,
                    "candidate": candidate,
                    "trades": int(len(group)),
                    "orderbook_covered_trades": int(covered.sum()),
                    "orderbook_coverage_rate": float(covered.mean()) if len(covered) else np.nan,
                    "avg_staleness_minutes": float(_num(group[covered], "orderbook_staleness_minutes").mean()) if covered.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _coverage_audit(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=COVERAGE_AUDIT_COLUMNS)
    rows = []
    for row in data.itertuples(index=False):
        payload = row._asdict()
        rows.append(
            {
                "event_id": payload.get("event_id", ""),
                "trade_id": payload.get("trade_id", ""),
                "signal_id": payload.get("signal_id", ""),
                "candidate": payload.get("candidate", ""),
                "pool_hint": payload.get("candidate_role", ""),
                "symbol": payload.get("symbol", ""),
                "decision_time": payload.get("decision_time", payload.get("entry_time")),
                "snapshot_time": payload.get("snapshot_time"),
                "snapshot_cts": payload.get("snapshot_cts"),
                "latency_ms": payload.get("latency_ms", np.nan),
                "snapshot_age_ms": payload.get("snapshot_age_ms", np.nan),
                "receive_delay_ms": payload.get("receive_delay_ms", np.nan),
                "coverage_status": payload.get("coverage_status", "unknown"),
                "levels_available": payload.get("levels_available", np.nan),
                "bid_levels_available": payload.get("bid_levels_available", np.nan),
                "ask_levels_available": payload.get("ask_levels_available", np.nan),
                "best_bid": payload.get("best_bid", np.nan),
                "best_ask": payload.get("best_ask", np.nan),
                "spread_bps": payload.get("spread_bps", np.nan),
                "source": payload.get("source", ""),
            }
        )
    audit = pd.DataFrame(rows, columns=COVERAGE_AUDIT_COLUMNS)
    return audit.sort_values(["decision_time", "symbol", "candidate"], na_position="last").reset_index(drop=True)


def _feature_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    covered = data[data.get("orderbook_covered", pd.Series(False, index=data.index)).fillna(False).astype(bool)].copy()
    for feature in ORDERBOOK_FEATURES:
        values = _num(covered, feature)
        rows.append(
            {
                "feature": feature,
                "covered_trades": int(len(covered)),
                "non_null": int(values.notna().sum()),
                "coverage_rate": float(values.notna().mean()) if len(values) else np.nan,
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "median": float(values.median()) if values.notna().any() else np.nan,
                "p10": float(values.quantile(0.10)) if values.notna().any() else np.nan,
                "p90": float(values.quantile(0.90)) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bucket_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        sample = sample[sample.get("orderbook_covered", pd.Series(False, index=sample.index)).fillna(False).astype(bool)].copy()
        if sample.empty:
            continue
        for feature in ORDERBOOK_FEATURES:
            values = _num(sample, feature)
            valid = sample[values.notna()].copy()
            if len(valid) < 10:
                continue
            try:
                valid["bucket"] = pd.qcut(_num(valid, feature), q=min(5, len(valid)), duplicates="drop")
            except ValueError:
                continue
            for bucket, group in valid.groupby("bucket", observed=True):
                rows.append(
                    {
                        "pool": pool_name,
                        "feature": feature,
                        "bucket": str(bucket),
                        "trades": int(len(group)),
                        "net20_avg": float(_num(group, "net_return").mean()),
                        "net20_sum": float(_num(group, "net_return").sum()),
                        "positive_rate": float((_num(group, "net_return") > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _ranking_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_frames = []
    skipped_frames = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        sample = sample[sample.get("orderbook_covered", pd.Series(False, index=sample.index)).fillna(False).astype(bool)].copy()
        if sample.empty:
            continue
        for rule, score_col in RANKING_RULES.items():
            if score_col not in sample.columns:
                continue
            for max_positions in MAX_POSITIONS:
                selected, skipped = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                rows.append(_metrics(selected, skipped, pool=pool_name, rule=rule, max_positions=max_positions))
                if pool_name == "P2_CIC1_CIC2_COMBINED" and max_positions in [5, 8]:
                    if not selected.empty:
                        selected_frames.append(selected.assign(ranking_rule=rule, pool=pool_name, max_positions=max_positions))
                    if not skipped.empty:
                        skipped_frames.append(skipped.assign(ranking_rule=rule, pool=pool_name, max_positions=max_positions))
    return (
        pd.DataFrame(rows, columns=[col for col in RANKING_COLUMNS if col != "random_percentile"]),
        pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
    )


def _random_comparison(data: pd.DataFrame, permutations: int = 100, seed: int = 909) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        sample = sample[sample.get("orderbook_covered", pd.Series(False, index=sample.index)).fillna(False).astype(bool)].copy()
        if sample.empty:
            continue
        distributions: dict[int, np.ndarray] = {}
        for max_positions in MAX_POSITIONS:
            totals = []
            for _ in range(permutations):
                local = sample.copy()
                local["rank_random"] = rng.normal(size=len(local))
                selected, _ = select_portfolio(local, score_col="rank_random", max_positions=max_positions)
                totals.append(float(_num(selected, "net_return").sum()) if not selected.empty else 0.0)
            distributions[max_positions] = np.array(totals, dtype=float)
        for rule, score_col in RANKING_RULES.items():
            if score_col not in sample.columns:
                continue
            for max_positions in MAX_POSITIONS:
                selected, _ = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                total = float(_num(selected, "net_return").sum()) if not selected.empty else 0.0
                dist = distributions[max_positions]
                rows.append(
                    {
                        "pool": pool_name,
                        "ranking_rule": rule,
                        "max_positions": max_positions,
                        "portfolio_net20": total,
                        "random_mean": float(np.mean(dist)),
                        "random_median": float(np.median(dist)),
                        "random_p75": float(np.quantile(dist, 0.75)),
                        "random_p90": float(np.quantile(dist, 0.90)),
                        "random_percentile": float((dist <= total).mean()),
                        "permutations": permutations,
                    }
                )
    return pd.DataFrame(rows)


def _conflict_set_analysis(data: pd.DataFrame, permutations: int = 50, seed: int = 910) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pool_name in POOLS:
        sample = _pool(data, pool_name)
        sample = sample[sample.get("orderbook_covered", pd.Series(False, index=sample.index)).fillna(False).astype(bool)].copy()
        if sample.empty:
            continue
        sample["decision_bucket"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce").dt.floor("15min")
        for rule, score_col in RANKING_RULES.items():
            if score_col not in sample.columns:
                continue
            for max_positions in MAX_POSITIONS:
                selected, skipped = select_portfolio(sample, score_col=score_col, max_positions=max_positions)
                joined = pd.concat(
                    [
                        selected.assign(selection_status="selected"),
                        skipped.assign(selection_status="skipped"),
                    ],
                    ignore_index=True,
                )
                if joined.empty:
                    continue
                joined["decision_bucket"] = pd.to_datetime(joined["entry_time"], utc=True, errors="coerce").dt.floor("15min")
                for bucket, group in joined.groupby("decision_bucket", sort=False, dropna=False):
                    if len(group) <= 1 or not group["selection_status"].eq("skipped").any():
                        continue
                    selected_group = group[group["selection_status"].eq("selected")]
                    skipped_group = group[group["selection_status"].eq("skipped")]
                    selected_count = min(max_positions, len(group))
                    oracle = _num(group, "net_return").sort_values(ascending=False).head(selected_count).sum()
                    random_totals = []
                    returns = _num(group, "net_return").to_numpy(dtype=float)
                    for _ in range(permutations):
                        if len(returns) <= selected_count:
                            random_totals.append(float(np.nansum(returns)))
                        else:
                            take = rng.choice(len(returns), size=selected_count, replace=False)
                            random_totals.append(float(np.nansum(returns[take])))
                    selected_total = float(_num(selected_group, "net_return").sum()) if not selected_group.empty else 0.0
                    rows.append(
                        {
                            "pool": pool_name,
                            "ranking_rule": rule,
                            "max_positions": max_positions,
                            "conflict_set_id": f"{pool_name}|{rule}|{max_positions}|{bucket}",
                            "decision_time": bucket,
                            "available_slots_proxy": int(max_positions),
                            "num_candidates": int(len(group)),
                            "selected_count": int(len(selected_group)),
                            "skipped_count": int(len(skipped_group)),
                            "selected_net20": float(_num(selected_group, "net_return").mean()) if len(selected_group) else np.nan,
                            "skipped_net20": float(_num(skipped_group, "net_return").mean()) if len(skipped_group) else np.nan,
                            "oracle_topk_net20": float(oracle),
                            "random_topk_mean_net20": float(np.mean(random_totals)),
                            "ranker_regret_vs_oracle": float(oracle - selected_total),
                            "ranker_lift_vs_random": float(selected_total - np.mean(random_totals)),
                        }
                    )
    return pd.DataFrame(rows)


def _book_quality_negative_controls(data: pd.DataFrame, seed: int = 911) -> pd.DataFrame:
    rows = []
    covered = data[data.get("orderbook_covered", pd.Series(False, index=data.index)).fillna(False).astype(bool)].copy()
    if covered.empty:
        return pd.DataFrame()
    controls = {
        "bad_wide_spread": _num(covered, "spread_bps") >= _num(covered, "spread_bps").quantile(0.80),
        "bad_low_bid_support": _num(covered, "bid_depth_25bp") <= _num(covered, "bid_depth_25bp").quantile(0.20),
        "bad_high_downside_risk": _num(covered, "downside_liquidity_risk_25bp")
        >= _num(covered, "downside_liquidity_risk_25bp").quantile(0.80),
        "bad_thick_ask_wall": _num(covered, "ask_depth_25bp") >= _num(covered, "ask_depth_25bp").quantile(0.80),
    }
    for name, mask in controls.items():
        group = covered[mask.fillna(False)]
        rows.append(
            {
                "control": name,
                "trades": int(len(group)),
                "net20_avg": float(_num(group, "net_return").mean()) if len(group) else np.nan,
                "net20_sum": float(_num(group, "net_return").sum()) if len(group) else 0.0,
                "positive_rate": float((_num(group, "net_return") > 0).mean()) if len(group) else np.nan,
            }
        )
    rng = np.random.default_rng(seed)
    shuffled = covered.copy()
    for feature in ["spread_bps", "bid_depth_25bp", "ask_depth_25bp", "imbalance_25bp", "entry_book_quality_score"]:
        if feature in shuffled.columns and len(shuffled) > 1:
            shuffled[f"random_snapshot_{feature}"] = rng.permutation(_num(shuffled, feature).to_numpy(dtype=float))
    rows.append(
        {
            "control": "random_snapshot_feature_permutation",
            "trades": int(len(shuffled)),
            "net20_avg": float(_num(shuffled, "net_return").mean()) if len(shuffled) else np.nan,
            "net20_sum": float(_num(shuffled, "net_return").sum()) if len(shuffled) else 0.0,
            "positive_rate": float((_num(shuffled, "net_return") > 0).mean()) if len(shuffled) else np.nan,
        }
    )
    future_like = data[data.get("coverage_status", pd.Series("", index=data.index)).astype(str).eq("future_only_snapshot")]
    rows.append(
        {
            "control": "delayed_or_future_snapshot_audit_only",
            "trades": int(len(future_like)),
            "net20_avg": float(_num(future_like, "net_return").mean()) if len(future_like) else np.nan,
            "net20_sum": float(_num(future_like, "net_return").sum()) if len(future_like) else 0.0,
            "positive_rate": float((_num(future_like, "net_return") > 0).mean()) if len(future_like) else np.nan,
        }
    )
    return pd.DataFrame(rows)


def _write_notes(path: Path, data: pd.DataFrame, coverage: pd.DataFrame, ranking: pd.DataFrame) -> None:
    cic = data[data["candidate"].isin(POOLS["P2_CIC1_CIC2_COMBINED"])] if not data.empty else data
    covered = int(cic.get("orderbook_covered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not cic.empty else 0
    if covered >= MIN_DECISION_TRADES:
        status = "candidate_check"
    elif covered >= MIN_DIAGNOSTIC_TRADES:
        status = "diagnostic_only"
    else:
        status = "insufficient_orderbook_coverage"
    lines = [
        "# v0.9E Orderbook-Aware Capacity Ranking",
        "",
        "phase: Coverage & As-of Audit first",
        f"sample_status: {status}",
        f"cic_pool_trades: {len(cic)}",
        f"entry_pre_orderbook_covered_trades: {covered}",
        "",
        "This report is shadow-only. It does not change CIC/P2 primary execution and does not allow real-live.",
        "",
        "Ranking uses only orderbook snapshots whose coverage status is `covered` or `covered_live_delay`.",
        "Coverage requires as-of snapshot timing, non-empty bid/ask, enough levels, sane spread, and staleness within tolerance.",
        "",
        "Read `orderbook_coverage_audit.csv` before interpreting any ranking or PnL table.",
        "",
    ]
    if coverage.empty or covered < MIN_DIAGNOSTIC_TRADES:
        lines.append("Current orderbook coverage is too small for ranking conclusions. Treat all ranking rows as framework smoke tests only.")
    elif not ranking.empty:
        lines.append("## Best Shadow Rows")
        for row in ranking.sort_values("portfolio_net20", ascending=False).head(5).itertuples(index=False):
            lines.append(
                f"- {row.pool} {row.ranking_rule} max={row.max_positions}: "
                f"portfolio_net20={row.portfolio_net20:.4%}, selected_minus_skipped={row.selected_minus_skipped:.4%}."
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v09e_orderbook_capacity_ranking(
    source_root: Path = SOURCE_ROOT,
    orderbook_root: Path = ORDERBOOK_ROOT,
    report_root: Path = REPORT_ROOT,
    *,
    max_staleness_minutes: int = MAX_STALENESS_MINUTES,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    trades = _prepare_trades(_read_trades(source_root))
    symbols = sorted(trades.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    features = _load_orderbook_features(orderbook_root, symbols)
    data = _add_rank_scores(
        _attach_orderbook_features(trades, features, max_staleness_minutes=max_staleness_minutes)
    )
    coverage = _coverage(data)
    coverage_audit = _coverage_audit(data)
    feature_summary = _feature_summary(data)
    bucket = _bucket_summary(data)
    ranking, selected, skipped = _ranking_summary(data)
    random = _random_comparison(data) if len(data) >= 5 else pd.DataFrame()
    conflict = _conflict_set_analysis(data) if len(data) >= 5 else pd.DataFrame()
    negative_controls = _book_quality_negative_controls(data)
    if not ranking.empty and not random.empty:
        ranking = ranking.merge(
            random[["pool", "ranking_rule", "max_positions", "random_percentile"]],
            on=["pool", "ranking_rule", "max_positions"],
            how="left",
        )
    for col in RANKING_COLUMNS:
        if col not in ranking.columns:
            ranking[col] = np.nan
    ranking = ranking[RANKING_COLUMNS]
    outputs = {
        "orderbook_coverage_audit": report_root / "orderbook_coverage_audit.csv",
        "orderbook_feature_coverage": report_root / "orderbook_feature_coverage.csv",
        "orderbook_feature_summary": report_root / "orderbook_feature_summary.csv",
        "orderbook_bucket_summary": report_root / "orderbook_bucket_summary.csv",
        "orderbook_ranking_summary": report_root / "orderbook_ranking_summary.csv",
        "selected_vs_skipped_orderbook": report_root / "selected_vs_skipped_orderbook.csv",
        "conflict_set_orderbook_analysis": report_root / "conflict_set_orderbook_analysis.csv",
        "random_permutation_comparison": report_root / "random_permutation_comparison.csv",
        "book_quality_negative_controls": report_root / "book_quality_negative_controls.csv",
        "selected_trades": report_root / "selected_trades.csv",
        "skipped_trades": report_root / "skipped_trades.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage_audit.to_csv(outputs["orderbook_coverage_audit"], index=False)
    coverage.to_csv(outputs["orderbook_feature_coverage"], index=False)
    feature_summary.to_csv(outputs["orderbook_feature_summary"], index=False)
    bucket.to_csv(outputs["orderbook_bucket_summary"], index=False)
    ranking.to_csv(outputs["orderbook_ranking_summary"], index=False)
    ranking.to_csv(outputs["selected_vs_skipped_orderbook"], index=False)
    conflict.to_csv(outputs["conflict_set_orderbook_analysis"], index=False)
    random.to_csv(outputs["random_permutation_comparison"], index=False)
    negative_controls.to_csv(outputs["book_quality_negative_controls"], index=False)
    selected.to_csv(outputs["selected_trades"], index=False)
    skipped.to_csv(outputs["skipped_trades"], index=False)
    _write_notes(outputs["candidate_notes"], data, coverage, ranking)
    return outputs


__all__ = ["REPORT_ROOT", "write_v09e_orderbook_capacity_ranking"]
