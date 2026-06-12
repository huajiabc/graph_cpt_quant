from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.clients.bybit import BybitClient
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet, write_parquet


REPORT_ROOT = Path("reports/v0_8_orderflow_shadow")
ORDERFLOW_ROOT = Path("data/orderflow/v0_8/bybit")
SOURCE_REPORT_ROOT = Path("reports/v0_7d2_cic_mir1_paper_live")
LIVE_FEATURE_PATH = Path("data/live_v07d2/processed/v0_7d2_live_features.parquet")
DEMAND_QUEUE_PATH = Path("data/orderflow/demand_queue.parquet")


RAW_COLUMNS = [
    "exchange",
    "symbol",
    "timestamp",
    "execId",
    "price",
    "size",
    "turnover",
    "side",
]

AGG_COLUMNS = [
    "exchange",
    "symbol",
    "bar_open_time",
    "trade_count",
    "buy_trade_count",
    "sell_trade_count",
    "volume",
    "turnover",
    "buy_volume",
    "sell_volume",
    "buy_turnover",
    "sell_turnover",
    "taker_buy_ratio",
    "cvd_delta_volume",
    "cvd_delta_turnover",
    "buy_sell_imbalance",
    "large_trade_threshold",
    "large_buy_count",
    "large_sell_count",
    "large_buy_turnover",
    "large_sell_turnover",
]


WINDOWS = [
    "shock_bar",
    "pullback_window",
    "reclaim_bar",
    "entry_bar",
    "post_entry_15m",
    "post_entry_1h",
]

DEMAND_COLUMNS = [
    "demand_id",
    "strategy_id",
    "exchange",
    "symbol",
    "window_start",
    "window_end",
    "priority",
    "reason",
    "status",
    "source_preference",
    "source_file",
    "source_row_id",
    "trade_id",
    "signal_id",
    "candidate",
    "candidate_role",
    "baseline_kind",
    "selected",
    "skip_reason",
    "coverage_ratio",
    "source_quality",
    "created_at",
    "updated_at",
]


@dataclass(frozen=True)
class OrderflowRunConfig:
    report_root: Path = REPORT_ROOT
    orderflow_root: Path = ORDERFLOW_ROOT
    source_report_root: Path = SOURCE_REPORT_ROOT
    live_feature_path: Path = LIVE_FEATURE_PATH
    demand_queue_path: Path = DEMAND_QUEUE_PATH
    top_n: int = 50
    max_symbols: int | None = None
    recent_trade_limit: int = 1000
    retain_days: int = 7
    report_lookback_days: int = 7
    large_trade_quantile: float = 0.95
    core_reference_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _empty_raw() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_COLUMNS)


def _empty_agg() -> pd.DataFrame:
    return pd.DataFrame(columns=AGG_COLUMNS)


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_parquet(path)
    except Exception as exc:  # pragma: no cover - exact parquet backend errors vary.
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        quarantine_path = path.with_suffix(f"{path.suffix}.corrupt.{stamp}")
        try:
            path.replace(quarantine_path)
            print(
                f"[orderflow] quarantined corrupt parquet cache {path} -> {quarantine_path}: {exc}",
                flush=True,
            )
        except OSError:
            print(f"[orderflow] failed to read parquet cache {path}: {exc}", flush=True)
        return pd.DataFrame()


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _coerce_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _raw_cache_path(root: Path, symbol: str) -> Path:
    return root / "recent_trades" / f"{symbol}.parquet"


def _agg_cache_path(root: Path, symbol: str) -> Path:
    return root / "orderflow_1m" / f"{symbol}.parquet"


def normalize_trade_frame(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    if df.empty:
        return _empty_raw()
    out = df.copy()
    if "exchange" not in out.columns:
        out["exchange"] = "bybit"
    if "symbol" not in out.columns and symbol:
        out["symbol"] = symbol
    out["timestamp"] = _coerce_timestamp_series(out["timestamp"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["size"] = pd.to_numeric(out["size"], errors="coerce")
    out["turnover"] = pd.to_numeric(out.get("turnover", out["price"] * out["size"]), errors="coerce")
    out["side"] = out["side"].astype(str)
    if "execId" not in out.columns:
        out["execId"] = (
            out["timestamp"].astype(str)
            + ":"
            + out["price"].astype(str)
            + ":"
            + out["size"].astype(str)
            + ":"
            + out["side"].astype(str)
        )
    return (
        out[RAW_COLUMNS]
        .dropna(subset=["timestamp", "price", "size"])
        .drop_duplicates(["exchange", "symbol", "execId"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def aggregate_orderflow_1m(
    trades: pd.DataFrame,
    large_trade_quantile: float = 0.95,
) -> pd.DataFrame:
    if trades.empty:
        return _empty_agg()
    data = normalize_trade_frame(trades)
    if data.empty:
        return _empty_agg()
    data["bar_open_time"] = data["timestamp"].dt.floor("1min")
    side = data["side"].str.lower()
    is_buy = side.eq("buy")
    is_sell = side.eq("sell")
    threshold = float(data["turnover"].quantile(large_trade_quantile))
    data["buy_trade_count"] = is_buy.astype(int)
    data["sell_trade_count"] = is_sell.astype(int)
    data["buy_volume"] = np.where(is_buy, data["size"], 0.0)
    data["sell_volume"] = np.where(is_sell, data["size"], 0.0)
    data["buy_turnover"] = np.where(is_buy, data["turnover"], 0.0)
    data["sell_turnover"] = np.where(is_sell, data["turnover"], 0.0)
    large = data["turnover"] >= threshold
    data["large_buy_count"] = (large & is_buy).astype(int)
    data["large_sell_count"] = (large & is_sell).astype(int)
    data["large_buy_turnover"] = np.where(large & is_buy, data["turnover"], 0.0)
    data["large_sell_turnover"] = np.where(large & is_sell, data["turnover"], 0.0)
    grouped = (
        data.groupby(["exchange", "symbol", "bar_open_time"], as_index=False)
        .agg(
            trade_count=("execId", "count"),
            buy_trade_count=("buy_trade_count", "sum"),
            sell_trade_count=("sell_trade_count", "sum"),
            volume=("size", "sum"),
            turnover=("turnover", "sum"),
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
            buy_turnover=("buy_turnover", "sum"),
            sell_turnover=("sell_turnover", "sum"),
            large_buy_count=("large_buy_count", "sum"),
            large_sell_count=("large_sell_count", "sum"),
            large_buy_turnover=("large_buy_turnover", "sum"),
            large_sell_turnover=("large_sell_turnover", "sum"),
        )
        .sort_values(["exchange", "symbol", "bar_open_time"])
        .reset_index(drop=True)
    )
    turnover = grouped["turnover"].replace(0, np.nan)
    grouped["taker_buy_ratio"] = grouped["buy_turnover"] / turnover
    grouped["cvd_delta_volume"] = grouped["buy_volume"] - grouped["sell_volume"]
    grouped["cvd_delta_turnover"] = grouped["buy_turnover"] - grouped["sell_turnover"]
    grouped["buy_sell_imbalance"] = grouped["cvd_delta_turnover"] / turnover
    grouped["large_trade_threshold"] = threshold
    return grouped[AGG_COLUMNS]


def summarize_trade_window(
    trades: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    large_trade_quantile: float = 0.95,
    *,
    assume_normalized: bool = False,
    large_trade_threshold: float | None = None,
) -> dict[str, object]:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return _empty_window_stats(start, end)
    start = _utc_timestamp(start)
    end = _utc_timestamp(end)
    if end <= start:
        end = start + pd.Timedelta(minutes=1)
    if trades.empty:
        return _empty_window_stats(start, end)
    data = trades if assume_normalized else normalize_trade_frame(trades)
    coverage_ratio = _coverage_ratio(data, start, end)
    cache_start = data["timestamp"].min() if not data.empty else pd.NaT
    cache_end = data["timestamp"].max() if not data.empty else pd.NaT
    sample = data[(data["timestamp"] >= start) & (data["timestamp"] < end)].copy()
    if sample.empty:
        stats = _empty_window_stats(start, end)
        stats.update(
            {
                "coverage_ratio": coverage_ratio,
                "source_quality": _source_quality(coverage_ratio, has_trades=False),
                "cache_start": cache_start,
                "cache_end": cache_end,
            }
        )
        return stats
    side = sample["side"].str.lower()
    is_buy = side.eq("buy")
    is_sell = side.eq("sell")
    if large_trade_threshold is not None and np.isfinite(large_trade_threshold):
        threshold = float(large_trade_threshold)
    else:
        threshold = float(data["turnover"].quantile(large_trade_quantile))
    large = sample["turnover"] >= threshold
    turnover = float(sample["turnover"].sum())
    buy_turnover = float(sample.loc[is_buy, "turnover"].sum())
    sell_turnover = float(sample.loc[is_sell, "turnover"].sum())
    buy_volume = float(sample.loc[is_buy, "size"].sum())
    sell_volume = float(sample.loc[is_sell, "size"].sum())
    return {
        "window_start": start,
        "window_end": end,
        "covered": True,
        "coverage_ratio": coverage_ratio,
        "source_quality": _source_quality(coverage_ratio, has_trades=True),
        "cache_start": cache_start,
        "cache_end": cache_end,
        "trade_count": int(len(sample)),
        "buy_trade_count": int(is_buy.sum()),
        "sell_trade_count": int(is_sell.sum()),
        "volume": float(sample["size"].sum()),
        "turnover": turnover,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "taker_buy_ratio": buy_turnover / turnover if turnover else np.nan,
        "cvd_delta_volume": buy_volume - sell_volume,
        "cvd_delta_turnover": buy_turnover - sell_turnover,
        "buy_sell_imbalance": (buy_turnover - sell_turnover) / turnover if turnover else np.nan,
        "large_trade_threshold": threshold,
        "large_buy_count": int((large & is_buy).sum()),
        "large_sell_count": int((large & is_sell).sum()),
        "large_buy_turnover": float(sample.loc[large & is_buy, "turnover"].sum()),
        "large_sell_turnover": float(sample.loc[large & is_sell, "turnover"].sum()),
    }


def _empty_window_stats(start: object, end: object) -> dict[str, object]:
    return {
        "window_start": start,
        "window_end": end,
        "covered": False,
        "coverage_ratio": 0.0,
        "source_quality": "missing",
        "cache_start": pd.NaT,
        "cache_end": pd.NaT,
        "trade_count": 0,
        "buy_trade_count": 0,
        "sell_trade_count": 0,
        "volume": 0.0,
        "turnover": 0.0,
        "buy_volume": 0.0,
        "sell_volume": 0.0,
        "buy_turnover": 0.0,
        "sell_turnover": 0.0,
        "taker_buy_ratio": np.nan,
        "cvd_delta_volume": np.nan,
        "cvd_delta_turnover": np.nan,
        "buy_sell_imbalance": np.nan,
        "large_trade_threshold": np.nan,
        "large_buy_count": 0,
        "large_sell_count": 0,
        "large_buy_turnover": 0.0,
        "large_sell_turnover": 0.0,
    }


def _coverage_ratio(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if trades.empty or pd.isna(start) or pd.isna(end) or end <= start:
        return 0.0
    cache_start = trades["timestamp"].min()
    cache_end = trades["timestamp"].max()
    if pd.isna(cache_start) or pd.isna(cache_end):
        return 0.0
    overlap_start = max(_utc_timestamp(start), _utc_timestamp(cache_start))
    overlap_end = min(_utc_timestamp(end), _utc_timestamp(cache_end))
    if overlap_end <= overlap_start:
        return 0.0
    window_seconds = max((end - start).total_seconds(), 1.0)
    return float(min(1.0, max(0.0, (overlap_end - overlap_start).total_seconds() / window_seconds)))


def _source_quality(coverage_ratio: float, *, has_trades: bool) -> str:
    if coverage_ratio >= 0.9 and has_trades:
        return "complete"
    if coverage_ratio >= 0.9:
        return "complete_empty"
    if coverage_ratio > 0:
        return "partial"
    return "missing"


def build_orderflow_demand_queue(
    source_report_root: Path = SOURCE_REPORT_ROOT,
    orderflow_root: Path = ORDERFLOW_ROOT,
    demand_queue_path: Path = DEMAND_QUEUE_PATH,
    *,
    report_lookback_days: int = 7,
    core_reference_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build an event-window demand queue for orderflow collection.

    The queue is intentionally event-driven. It prioritizes actual paper-live
    and shadow trade windows over broad TopN caching so small recorder budgets
    cover the symbols needed for CIC/MIR capacity and ranking diagnostics.
    """

    created_at = _utc_timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    cutoff = created_at - pd.Timedelta(days=report_lookback_days)
    rows: list[dict[str, object]] = []
    source_specs = [
        ("paper_trades.parquet", "paper_trade"),
        ("paper_portfolio_trades.parquet", "shadow_portfolio_trade"),
        ("paper_skipped_candidates.parquet", "shadow_skipped_candidate"),
        ("baseline_trades.parquet", "baseline_trade"),
    ]
    for filename, source_kind in source_specs:
        path = source_report_root / filename
        if not path.exists():
            continue
        df = read_parquet(path)
        if df.empty or "symbol" not in df.columns:
            continue
        df = _recent_event_rows(df, cutoff)
        for idx, row in df.reset_index(drop=True).iterrows():
            demand = _demand_row_from_event(row, source_kind, filename, int(idx), created_at, orderflow_root)
            if demand:
                rows.append(demand)

    signal_path = source_report_root / "paper_signals.parquet"
    if signal_path.exists():
        signals = read_parquet(signal_path)
        if not signals.empty and "symbol" in signals.columns:
            signals = _recent_event_rows(signals, cutoff)
            interesting = signals[
                signals.get("status", pd.Series("", index=signals.index)).astype(str).isin(
                    ["detected", "armed", "pullback_seen", "reclaim_seen"]
                )
            ].copy()
            for idx, row in interesting.reset_index(drop=True).iterrows():
                demand = _demand_row_from_event(row, "paper_signal", "paper_signals.parquet", int(idx), created_at, orderflow_root)
                if demand:
                    rows.append(demand)

    for symbol in core_reference_symbols:
        rows.append(
            _finalize_demand_row(
                {
                    "strategy_id": "core_reference_recorder",
                    "exchange": "bybit",
                    "symbol": symbol,
                    "window_start": created_at - pd.Timedelta(hours=2),
                    "window_end": created_at + pd.Timedelta(minutes=5),
                    "priority": "P4",
                    "reason": "core_reference_symbol",
                    "source_preference": "bybit_recent_trades_rest",
                    "source_file": "core_reference",
                    "source_row_id": -1,
                    "trade_id": "",
                    "signal_id": "",
                    "candidate": "CORE_REFERENCE",
                    "candidate_role": "reference",
                    "baseline_kind": "",
                    "selected": False,
                    "skip_reason": "",
                    "created_at": created_at,
                    "updated_at": created_at,
                },
                orderflow_root,
            )
        )

    queue = pd.DataFrame(rows, columns=DEMAND_COLUMNS)
    if not queue.empty:
        queue = (
            queue.drop_duplicates(["exchange", "symbol", "window_start", "window_end", "reason"])
            .sort_values(["priority", "window_end", "symbol"], ascending=[True, False, True])
            .reset_index(drop=True)
        )
        queue["demand_id"] = [
            _demand_id(row) for _, row in queue.iterrows()
        ]
    else:
        queue = pd.DataFrame(columns=DEMAND_COLUMNS)
    write_parquet(queue, demand_queue_path)
    return queue


def _recent_event_rows(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    data = df.copy()
    time_cols = [
        "entry_time",
        "reclaim_time",
        "pullback_time",
        "local_volume_shock_time",
        "bar_close_time",
        "feature_time",
    ]
    mask = pd.Series(False, index=data.index)
    found = False
    for col in time_cols:
        if col not in data.columns:
            continue
        found = True
        mask |= _coerce_timestamp_series(data[col]) >= cutoff
    return data[mask].copy() if found else data.copy()


def _demand_row_from_event(
    row: pd.Series,
    source_kind: str,
    source_file: str,
    source_row_id: int,
    created_at: pd.Timestamp,
    orderflow_root: Path,
) -> dict[str, object] | None:
    symbol = str(row.get("symbol", "") or "")
    if not symbol:
        return None
    window_start, window_end = _event_demand_window(row)
    if window_start is None or window_end is None or window_end <= window_start:
        return None
    selected = bool(row.get("selected", row.get("portfolio_accepted", False)))
    skip_reason = str(row.get("skip_reason", row.get("portfolio_skip_reason", "")) or "")
    candidate = str(row.get("candidate", "") or "")
    role = str(row.get("candidate_role", "") or "")
    baseline = str(row.get("baseline_kind", "") or "")
    priority, reason = _demand_priority_reason(source_kind, candidate, role, baseline, selected, skip_reason)
    return _finalize_demand_row(
        {
            "strategy_id": "v0_8_1_orderflow_demand_queue",
            "exchange": str(row.get("exchange", "bybit") or "bybit"),
            "symbol": symbol,
            "window_start": window_start,
            "window_end": window_end,
            "priority": priority,
            "reason": reason,
            "source_preference": "bybit_recent_trades_rest|bybit_archive|tardis|binance_proxy",
            "source_file": source_file,
            "source_row_id": source_row_id,
            "trade_id": str(row.get("trade_id", "") or ""),
            "signal_id": str(row.get("signal_id", "") or ""),
            "candidate": candidate,
            "candidate_role": role,
            "baseline_kind": baseline,
            "selected": selected,
            "skip_reason": skip_reason,
            "created_at": created_at,
            "updated_at": created_at,
        },
        orderflow_root,
    )


def _event_demand_window(row: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    times = [
        _maybe_time(row.get("local_volume_shock_time")),
        _maybe_time(row.get("pullback_time")),
        _maybe_time(row.get("reclaim_time")),
        _maybe_time(row.get("entry_time")),
    ]
    valid_times = [time for time in times if time is not None]
    start_anchor = min(valid_times) if valid_times else _maybe_time(row.get("bar_close_time"))
    if start_anchor is None:
        start_anchor = _maybe_time(row.get("feature_time"))
    if start_anchor is None:
        return (None, None)
    entry = _maybe_time(row.get("entry_time"))
    exit_time = _maybe_time(row.get("exit_time"))
    entry_valid_until = _maybe_time(row.get("entry_valid_until"))
    if entry is not None:
        fallback_end = entry + pd.Timedelta(hours=2)
        if exit_time is not None:
            fallback_end = min(exit_time + pd.Timedelta(minutes=30), fallback_end)
        end = fallback_end
    elif entry_valid_until is not None:
        end = entry_valid_until + pd.Timedelta(minutes=30)
    else:
        end = start_anchor + pd.Timedelta(hours=2)
    return (start_anchor - pd.Timedelta(minutes=30), end)


def _demand_priority_reason(
    source_kind: str,
    candidate: str,
    role: str,
    baseline: str,
    selected: bool,
    skip_reason: str,
) -> tuple[str, str]:
    candidate_upper = candidate.upper()
    if selected and candidate_upper == "CIC1_FILTERED_MIR1":
        return ("P0", "cic_primary_selected_trade")
    if "portfolio_full" in skip_reason or "no_capacity" in skip_reason or "lower_rank" in skip_reason:
        return ("P1", "capacity_skipped_candidate")
    if source_kind == "paper_signal":
        return ("P2", "active_or_recent_signal")
    if source_kind in {"shadow_portfolio_trade", "shadow_skipped_candidate"}:
        return ("P2", f"{source_kind}")
    if candidate_upper in {"CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"} and not baseline:
        return ("P2", "cic_candidate_trade")
    if candidate_upper in {"MIR1_RAW", "IR2_REF"} or baseline or role in {"reference", "legacy_reference"}:
        return ("P3", "reference_or_baseline_trade")
    return ("P3", source_kind)


def _finalize_demand_row(row: dict[str, object], orderflow_root: Path) -> dict[str, object]:
    symbol = str(row["symbol"])
    raw = normalize_trade_frame(_read_optional_parquet(_raw_cache_path(orderflow_root, symbol)), symbol)
    start = _maybe_time(row.get("window_start"))
    end = _maybe_time(row.get("window_end"))
    ratio = _coverage_ratio(raw, start, end) if start is not None and end is not None else 0.0
    quality = _source_quality(ratio, has_trades=not raw.empty)
    row["coverage_ratio"] = ratio
    row["source_quality"] = quality
    row["status"] = "done" if ratio >= 0.9 else ("partial" if ratio > 0 else "pending")
    return row


def _demand_id(row: pd.Series) -> str:
    start = _maybe_time(row.get("window_start"))
    start_text = start.strftime("%Y%m%dT%H%M%SZ") if start is not None else "unknown"
    return (
        f"{row.get('priority')}:{row.get('exchange')}:{row.get('symbol')}:"
        f"{start_text}:{row.get('reason')}:{row.name}"
    )


def select_orderflow_symbols(
    live_feature_path: Path,
    source_report_root: Path,
    top_n: int = 50,
    report_lookback_days: int = 7,
    max_symbols: int | None = None,
    demand_queue: pd.DataFrame | None = None,
) -> list[str]:
    feature_symbols: list[str] = []
    report_symbols: list[str] = []
    demand_symbols: list[str] = []
    if demand_queue is not None and not demand_queue.empty and "symbol" in demand_queue.columns:
        queue = demand_queue.copy()
        if "priority" not in queue.columns:
            queue["priority"] = "P3"
        if "status" not in queue.columns:
            queue["status"] = "pending"
        status_order = {"pending": 0, "partial": 1, "failed_network": 2, "done": 3}
        queue["_status_order"] = queue["status"].astype(str).map(status_order).fillna(2)
        queue = queue.sort_values(
            ["priority", "_status_order", "window_end", "symbol"],
            ascending=[True, True, False, True],
        )
        demand_symbols.extend(queue["symbol"].dropna().astype(str).tolist())
    if live_feature_path.exists():
        features = read_parquet(live_feature_path)
        if not features.empty and "symbol" in features.columns:
            rank_col = "dynamic_all_rank" if "dynamic_all_rank" in features.columns else "turnover_rank_30d"
            data = features.copy()
            data[rank_col] = pd.to_numeric(data.get(rank_col), errors="coerce")
            time_col = "feature_time" if "feature_time" in data.columns else "bar_close_time"
            if time_col in data.columns:
                data[time_col] = _coerce_timestamp_series(data[time_col])
                latest = data[time_col].max()
                data = data[data[time_col] >= latest - pd.Timedelta(days=7)]
            ranked = (
                data[data[rank_col] <= top_n]
                .groupby("symbol", as_index=False)[rank_col]
                .min()
                .sort_values([rank_col, "symbol"])
            )
            feature_symbols.extend(ranked["symbol"].dropna().astype(str).tolist())
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=report_lookback_days)
    for filename in ["paper_signals.parquet", "paper_trades.parquet", "baseline_trades.parquet"]:
        path = source_report_root / filename
        if not path.exists():
            continue
        df = read_parquet(path)
        if df.empty or "symbol" not in df.columns:
            continue
        time_cols = [col for col in ["entry_time", "reclaim_time", "local_volume_shock_time"] if col in df.columns]
        if time_cols:
            mask = pd.Series(False, index=df.index)
            for col in time_cols:
                mask |= _coerce_timestamp_series(df[col]) >= cutoff
            df = df[mask]
        report_symbols.extend(df["symbol"].dropna().astype(str).tolist())
    # Paper-trade symbols are prioritized so a small max_symbols cap does not
    # spend the whole cache budget on liquid leaders unrelated to current paths.
    unique = list(dict.fromkeys(demand_symbols + report_symbols + feature_symbols))
    return unique[:max_symbols] if max_symbols else unique


def update_recent_trade_cache(
    client: BybitClient,
    symbols: list[str],
    orderflow_root: Path,
    recent_trade_limit: int = 1000,
    retain_days: int = 7,
    large_trade_quantile: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw_dir = ensure_dir(orderflow_root / "recent_trades")
    agg_dir = ensure_dir(orderflow_root / "orderflow_1m")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=retain_days)
    raw_by_symbol: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    fetched_at = pd.Timestamp.now(tz="UTC")
    for symbol in symbols:
        path = raw_dir / f"{symbol}.parquet"
        cached = normalize_trade_frame(_read_optional_parquet(path), symbol)
        error = ""
        fetched = _empty_raw()
        try:
            fetched = normalize_trade_frame(client.recent_trades(symbol, recent_trade_limit), symbol)
        except Exception as exc:  # pragma: no cover - live network failure only
            error = str(exc)
        combined = pd.concat([cached, fetched], ignore_index=True) if not cached.empty else fetched
        combined = normalize_trade_frame(combined, symbol)
        if not combined.empty:
            combined = combined[combined["timestamp"] >= cutoff].copy()
        write_parquet(combined, path)
        agg = aggregate_orderflow_1m(combined, large_trade_quantile)
        write_parquet(agg, agg_dir / f"{symbol}.parquet")
        raw_by_symbol[symbol] = combined
        rows.append(
            {
                "symbol": symbol,
                "fetched_at": fetched_at,
                "new_rows": int(len(fetched)),
                "cached_rows": int(len(combined)),
                "cached_start": combined["timestamp"].min() if not combined.empty else pd.NaT,
                "cached_end": combined["timestamp"].max() if not combined.empty else pd.NaT,
                "orderflow_1m_rows": int(len(agg)),
                "error": error,
            }
        )
    return pd.DataFrame(rows), raw_by_symbol


def _trade_windows(row: pd.Series) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    shock = _maybe_time(row.get("local_volume_shock_time"))
    pullback = _maybe_time(row.get("pullback_time"))
    reclaim = _maybe_time(row.get("reclaim_time"))
    entry = _maybe_time(row.get("entry_time"))
    return {
        "shock_bar": _bar_window(shock, minutes=15),
        "pullback_window": (pullback, reclaim if reclaim and pullback and reclaim > pullback else None),
        "reclaim_bar": _bar_window(reclaim, minutes=15),
        "entry_bar": _bar_window(entry, minutes=15),
        "post_entry_15m": _bar_window(entry, minutes=15),
        "post_entry_1h": _bar_window(entry, minutes=60),
    }


def _maybe_time(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return _utc_timestamp(value)


def _bar_window(start: pd.Timestamp | None, minutes: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if start is None:
        return (None, None)
    return (start, start + pd.Timedelta(minutes=minutes))


def attach_orderflow_to_trades(
    trades: pd.DataFrame,
    raw_by_symbol: dict[str, pd.DataFrame],
    large_trade_quantile: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), pd.DataFrame()
    flat_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for _, trade in trades.iterrows():
        symbol = str(trade.get("symbol", ""))
        raw = raw_by_symbol.get(symbol, _empty_raw())
        flat = trade.to_dict()
        for window_name, (start, end) in _trade_windows(trade).items():
            if window_name == "pullback_window" and (end is None) and start is not None:
                end = start + pd.Timedelta(minutes=15)
            stats = summarize_trade_window(raw, start, end, large_trade_quantile)
            long_row = {
                "trade_id": trade.get("trade_id"),
                "signal_id": trade.get("signal_id"),
                "candidate": trade.get("candidate"),
                "candidate_role": trade.get("candidate_role"),
                "baseline_kind": trade.get("baseline_kind", ""),
                "portfolio_accepted": trade.get("portfolio_accepted", False),
                "exchange": trade.get("exchange"),
                "symbol": symbol,
                "window": window_name,
                **stats,
            }
            window_rows.append(long_row)
            for key, value in stats.items():
                flat[f"{window_name}_{key}"] = value
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows), pd.DataFrame(window_rows)


def orderflow_confirmation_summary(shadow_trades: pd.DataFrame) -> pd.DataFrame:
    if shadow_trades.empty:
        return pd.DataFrame()
    data = shadow_trades.copy()
    keys = ["candidate", "candidate_role", "baseline_kind", "portfolio_accepted"]
    for col in keys:
        if col not in data.columns:
            data[col] = ""
    data["reclaim_covered"] = data.get("reclaim_bar_covered", False).fillna(False).astype(bool)
    data["entry_covered"] = data.get("entry_bar_covered", False).fillna(False).astype(bool)
    data["post_entry_covered"] = data.get("post_entry_15m_covered", False).fillna(False).astype(bool)
    data["reclaim_buy_confirmed"] = pd.to_numeric(
        data.get("reclaim_bar_taker_buy_ratio"), errors="coerce"
    ) >= 0.60
    data["reclaim_cvd_positive"] = pd.to_numeric(
        data.get("reclaim_bar_cvd_delta_turnover"), errors="coerce"
    ) > 0
    data["post_entry_cvd_positive"] = pd.to_numeric(
        data.get("post_entry_15m_cvd_delta_turnover"), errors="coerce"
    ) > 0
    grouped = data.groupby(keys, dropna=False)
    rows = []
    for group_key, sample in grouped:
        row = dict(zip(keys, group_key))
        row.update(
            {
                "trades": int(len(sample)),
                "portfolio_trades": int(sample["portfolio_accepted"].fillna(False).sum()),
                "reclaim_coverage_rate": float(sample["reclaim_covered"].mean()),
                "entry_coverage_rate": float(sample["entry_covered"].mean()),
                "post_entry_15m_coverage_rate": float(sample["post_entry_covered"].mean()),
                "reclaim_taker_buy_ratio_avg": _mean(sample, "reclaim_bar_taker_buy_ratio"),
                "reclaim_buy_confirmed_rate": _covered_mean(
                    sample, "reclaim_buy_confirmed", "reclaim_covered"
                ),
                "reclaim_cvd_positive_rate": _covered_mean(
                    sample, "reclaim_cvd_positive", "reclaim_covered"
                ),
                "post_entry_cvd_positive_rate": _covered_mean(
                    sample, "post_entry_cvd_positive", "post_entry_covered"
                ),
                "net10_avg": _mean(sample, "net_return_10bp"),
                "net20_avg": _mean(sample, "net_return_20bp"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _covered_mean(df: pd.DataFrame, value_col: str, covered_col: str) -> float:
    covered = df[covered_col].fillna(False).astype(bool)
    if not covered.any():
        return float("nan")
    return float(df.loc[covered, value_col].fillna(False).astype(bool).mean())


def write_v08_orderflow_shadow(
    base_config: ExperimentConfig,
    run_config: OrderflowRunConfig | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Path]:
    cfg = run_config or OrderflowRunConfig()
    report_root = ensure_dir(cfg.report_root)
    orderflow_root = ensure_dir(cfg.orderflow_root)
    demand_queue = build_orderflow_demand_queue(
        cfg.source_report_root,
        orderflow_root,
        cfg.demand_queue_path,
        report_lookback_days=cfg.report_lookback_days,
        core_reference_symbols=cfg.core_reference_symbols,
    )
    selected = symbols or select_orderflow_symbols(
        cfg.live_feature_path,
        cfg.source_report_root,
        cfg.top_n,
        cfg.report_lookback_days,
        cfg.max_symbols,
        demand_queue,
    )
    client = BybitClient(
        str(base_config.exchanges.bybit.base_url),
        base_config.exchanges.bybit.category,
    )
    try:
        cache_status, raw_by_symbol = update_recent_trade_cache(
            client,
            selected,
            orderflow_root,
            cfg.recent_trade_limit,
            cfg.retain_days,
            cfg.large_trade_quantile,
        )
    finally:
        client.close()
    demand_queue = build_orderflow_demand_queue(
        cfg.source_report_root,
        orderflow_root,
        cfg.demand_queue_path,
        report_lookback_days=cfg.report_lookback_days,
        core_reference_symbols=cfg.core_reference_symbols,
    )
    trade_path = cfg.source_report_root / "paper_trades.parquet"
    trades = read_parquet(trade_path) if trade_path.exists() else pd.DataFrame()
    missing_symbols = set(trades.get("symbol", pd.Series(dtype=str)).dropna().astype(str)) - set(raw_by_symbol)
    for symbol in sorted(missing_symbols):
        raw_by_symbol[symbol] = normalize_trade_frame(_read_optional_parquet(_raw_cache_path(orderflow_root, symbol)), symbol)
    shadow_trades, window_summary = attach_orderflow_to_trades(
        trades,
        raw_by_symbol,
        cfg.large_trade_quantile,
    )
    confirmation = orderflow_confirmation_summary(shadow_trades)
    outputs = {
        "cache_status": report_root / "orderflow_cache_status.csv",
        "demand_queue": cfg.demand_queue_path,
        "demand_queue_csv": report_root / "orderflow_demand_queue.csv",
        "shadow_trades": report_root / "orderflow_shadow_trades.csv",
        "window_summary": report_root / "orderflow_window_summary.csv",
        "confirmation_summary": report_root / "orderflow_confirmation_summary.csv",
        "current_status": report_root / "current_status.md",
    }
    cache_status.to_csv(outputs["cache_status"], index=False)
    demand_queue.to_csv(outputs["demand_queue_csv"], index=False)
    shadow_trades.to_csv(outputs["shadow_trades"], index=False)
    window_summary.to_csv(outputs["window_summary"], index=False)
    confirmation.to_csv(outputs["confirmation_summary"], index=False)
    _write_current_status(outputs["current_status"], selected, cache_status, shadow_trades, cfg, demand_queue)
    return outputs


def _write_current_status(
    path: Path,
    symbols: list[str],
    cache_status: pd.DataFrame,
    shadow_trades: pd.DataFrame,
    cfg: OrderflowRunConfig,
    demand_queue: pd.DataFrame | None = None,
) -> None:
    covered = (
        shadow_trades.get("reclaim_bar_covered", pd.Series(dtype=bool)).fillna(False).astype(bool)
        if not shadow_trades.empty
        else pd.Series(dtype=bool)
    )
    primary = (
        shadow_trades.get("candidate", pd.Series(dtype=str)).astype(str).eq("CIC1_FILTERED_MIR1")
        & shadow_trades.get("portfolio_accepted", pd.Series(dtype=bool)).fillna(False).astype(bool)
        if not shadow_trades.empty
        else pd.Series(dtype=bool)
    )
    errors = cache_status["error"].astype(str).ne("").sum() if "error" in cache_status.columns else 0
    latest_cache = cache_status["cached_end"].max() if "cached_end" in cache_status.columns else pd.NaT
    demand_counts = (
        demand_queue.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        if demand_queue is not None and not demand_queue.empty
        else {}
    )
    priority_counts = (
        demand_queue.get("priority", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        if demand_queue is not None and not demand_queue.empty
        else {}
    )
    demand_symbols = (
        int(demand_queue.get("symbol", pd.Series(dtype=str)).dropna().astype(str).nunique())
        if demand_queue is not None and not demand_queue.empty
        else 0
    )
    lines = [
        "# v0.8 Orderflow Shadow Status",
        "",
        "- mode: shadow_only",
        "- source: Bybit recent trades REST + v0.8.1 event demand queue",
        "- trading_decision_impact: none",
        "- real_live_allowed: false",
        "- historical_archive_tick_validation: pending",
        f"- selected_symbols: {len(symbols)}",
        f"- recent_trade_limit: {cfg.recent_trade_limit}",
        f"- retain_days: {cfg.retain_days}",
        f"- latest_cached_trade_time: {latest_cache}",
        f"- cache_errors: {int(errors)}",
        f"- demand_rows: {0 if demand_queue is None else len(demand_queue)}",
        f"- demand_symbols: {demand_symbols}",
        f"- demand_status_counts: {demand_counts}",
        f"- demand_priority_counts: {priority_counts}",
        f"- shadow_trades: {len(shadow_trades)}",
        f"- reclaim_window_coverage_rate: {float(covered.mean()):.2%}" if len(covered) else "- reclaim_window_coverage_rate: n/a",
        f"- primary_cic1_portfolio_trades: {int(primary.sum())}" if len(primary) else "- primary_cic1_portfolio_trades: 0",
        "",
        "Notes:",
        "- This module only records taker-flow context for paper trades and baselines.",
        "- v0.8.1 writes an event-driven demand queue so recorder budget follows selected/skipped CIC windows first.",
        "- Missing coverage is expected until the live tape cache has been running long enough.",
        "- Do not use this report for real-live permission without tick/orderflow validation and paper-live sample.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "ORDERFLOW_ROOT",
    "REPORT_ROOT",
    "SOURCE_REPORT_ROOT",
    "OrderflowRunConfig",
    "aggregate_orderflow_1m",
    "attach_orderflow_to_trades",
    "orderflow_confirmation_summary",
    "select_orderflow_symbols",
    "summarize_trade_window",
    "update_recent_trade_cache",
    "write_v08_orderflow_shadow",
]
