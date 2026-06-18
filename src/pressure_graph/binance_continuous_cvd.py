"""Continuous Binance UM aggTrades CVD backfill (77.docx P1, vData).

The v11 ``orderflow_history.py`` module ships CIC-anchored event
windows: shock_bar / pullback_window / reclaim_bar / entry_bar /
post_entry_1h. That gave the v7S Direction E sell-flow gate a usable
read at the CIC entry timestamp BUT not at the breakdown bar — only
574 CIC-anchored events in the parquet, vs the millions of bars in
the universe. Direction A's bootstrap CI straddled zero because of
this.

This module is the continuous-coverage counterpart. For each
(symbol, day) it:

1. Loads the cached aggTrades zip via ``orderflow_history.load_aggtrades_zip``
   (re-using the existing download + cache layer).
2. Bins the trades into fixed-size bars (default ``1m``, ``5m``, ``15m``).
3. Computes per-bin CVD features: ``taker_buy_ratio``, ``cvd_delta_volume``,
   ``cvd_delta_turnover``, ``buy_sell_imbalance``, ``large_buy_count``,
   ``large_sell_count``, ``trade_count``, ``coverage_ratio``,
   ``source_quality``.
4. Appends to a per-symbol, per-month parquet at
   ``data/orderflow_history/binance_um/continuous/<symbol>/<YYYY-MM>.parquet``.
5. Records the (symbol, day) outcome in the coverage report.

The output schema matches the v11 window aggregations one-for-one
(``shock_bar_taker_buy_ratio`` ↔ ``taker_buy_ratio``) so downstream
event lookups can swap the data source without touching feature
semantics.

Pure functions are unit-testable with synthetic trade frames. The
public ``backfill_symbol_days`` driver wraps them with IO and stats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.orderflow_history import (
    HISTORY_ROOT,
    OrderflowHistoryConfig,
    aggtrades_zip_path,
    download_aggtrades_day,
    load_aggtrades_zip,
)

CONTINUOUS_ROOT = HISTORY_ROOT / "continuous"
COVERAGE_REPORT_PATH = HISTORY_ROOT / "continuous_coverage.parquet"
QUALITY_AUDIT_PATH = HISTORY_ROOT / "continuous_quality_audit.parquet"

DEFAULT_BAR_SIZES: tuple[str, ...] = ("1min", "5min", "15min")
LARGE_TRADE_TURNOVER_QUANTILE: float = 0.95  # top 5 % of trade turnover per day = "large"


@dataclass(frozen=True)
class ContinuousCvdConfig:
    """Knobs for the continuous backfill."""

    history_root: Path = HISTORY_ROOT
    continuous_root: Path = CONTINUOUS_ROOT
    bar_sizes: tuple[str, ...] = DEFAULT_BAR_SIZES
    large_trade_turnover_quantile: float = LARGE_TRADE_TURNOVER_QUANTILE
    download_if_missing: bool = False  # set True on the server with proxy access
    timeout_seconds: int = 120
    download_workers: int = 4

    @property
    def orderflow_cfg(self) -> OrderflowHistoryConfig:
        return OrderflowHistoryConfig(
            history_root=self.history_root,
            timeout_seconds=self.timeout_seconds,
            download_workers=self.download_workers,
        )


@dataclass(frozen=True)
class DayOutcome:
    symbol: str
    day: date
    source_quality: str  # 'complete' | 'partial' | 'empty' | 'missing' | 'error'
    trade_count: int
    coverage_seconds: int
    error: str = ""


@dataclass
class BackfillStats:
    days_attempted: int = 0
    days_complete: int = 0
    days_empty: int = 0
    days_missing: int = 0
    days_error: int = 0
    bars_written: int = 0
    outcomes: list[DayOutcome] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Pure binning + feature computation
# --------------------------------------------------------------------------------------


def _expected_bar_count(day: date, bar_size: str) -> int:
    """Number of bars in a UTC day for the given bar size (e.g. 1440 for 1min)."""
    seconds_per_day = 86400
    freq_seconds = int(pd.Timedelta(bar_size).total_seconds())
    if freq_seconds <= 0:
        return 0
    return seconds_per_day // freq_seconds


def _empty_features(bar_size: str) -> pd.DataFrame:
    cols = [
        "symbol", "bar_open_time", "bar_size",
        "trade_count", "volume", "turnover",
        "buy_volume", "sell_volume", "buy_turnover", "sell_turnover",
        "taker_buy_ratio", "buy_sell_imbalance",
        "cvd_delta_volume", "cvd_delta_turnover",
        "large_trade_threshold",
        "large_buy_count", "large_sell_count",
        "large_buy_turnover", "large_sell_turnover",
        "coverage_ratio", "source_quality",
    ]
    return pd.DataFrame(columns=cols)


def compute_continuous_features(
    trades: pd.DataFrame,
    *,
    symbol: str,
    day: date,
    bar_size: str,
    large_quantile: float = LARGE_TRADE_TURNOVER_QUANTILE,
) -> pd.DataFrame:
    """Aggregate normalized trades into ``bar_size``-binned CVD features.

    ``trades`` must follow ``orderflow_history.normalize_aggtrades`` schema
    (timestamp, price, size, turnover, side). Empty input returns the
    empty schema. The returned frame has one row per bar within the UTC
    day; bars with zero trades get all-zero metrics tagged
    ``source_quality='empty'``.
    """
    if trades.empty or "timestamp" not in trades.columns:
        return _empty_features(bar_size)
    sub = trades.copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
    sub = sub.dropna(subset=["timestamp"])
    if sub.empty:
        return _empty_features(bar_size)

    day_start = pd.Timestamp(day, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    sub = sub[(sub["timestamp"] >= day_start) & (sub["timestamp"] < day_end)]
    if sub.empty:
        return _empty_features(bar_size)

    sub["is_buy"] = sub["side"].astype(str).str.lower() == "buy"
    turnover = pd.to_numeric(sub["turnover"], errors="coerce").fillna(0.0).to_numpy()
    if turnover.size:
        large_threshold = float(np.quantile(turnover[turnover > 0], large_quantile)) if (turnover > 0).any() else 0.0
    else:
        large_threshold = 0.0
    sub["is_large"] = pd.to_numeric(sub["turnover"], errors="coerce").fillna(0.0) >= large_threshold

    sub["bar_open_time"] = sub["timestamp"].dt.floor(bar_size)
    grouper = sub.groupby("bar_open_time", sort=True)
    out = pd.DataFrame(index=grouper.groups.keys())

    out["trade_count"] = grouper.size().astype(int)
    out["volume"] = grouper["size"].sum()
    out["turnover"] = grouper["turnover"].sum()
    buy_only = sub[sub["is_buy"]].groupby("bar_open_time")
    sell_only = sub[~sub["is_buy"]].groupby("bar_open_time")
    out["buy_volume"] = buy_only["size"].sum().reindex(out.index, fill_value=0.0)
    out["sell_volume"] = sell_only["size"].sum().reindex(out.index, fill_value=0.0)
    out["buy_turnover"] = buy_only["turnover"].sum().reindex(out.index, fill_value=0.0)
    out["sell_turnover"] = sell_only["turnover"].sum().reindex(out.index, fill_value=0.0)
    safe_vol = out["volume"].replace(0.0, np.nan)
    out["taker_buy_ratio"] = (out["buy_volume"] / safe_vol).fillna(0.5)
    out["buy_sell_imbalance"] = (out["buy_turnover"] - out["sell_turnover"]) / out["turnover"].replace(0.0, np.nan)
    out["buy_sell_imbalance"] = out["buy_sell_imbalance"].fillna(0.0)
    out["cvd_delta_volume"] = out["buy_volume"] - out["sell_volume"]
    out["cvd_delta_turnover"] = out["buy_turnover"] - out["sell_turnover"]
    out["large_trade_threshold"] = large_threshold
    large_buy_only = sub[sub["is_buy"] & sub["is_large"]].groupby("bar_open_time")
    large_sell_only = sub[~sub["is_buy"] & sub["is_large"]].groupby("bar_open_time")
    out["large_buy_count"] = large_buy_only.size().reindex(out.index, fill_value=0).astype(int)
    out["large_sell_count"] = large_sell_only.size().reindex(out.index, fill_value=0).astype(int)
    out["large_buy_turnover"] = large_buy_only["turnover"].sum().reindex(out.index, fill_value=0.0)
    out["large_sell_turnover"] = large_sell_only["turnover"].sum().reindex(out.index, fill_value=0.0)

    # Coverage ratio: distinct 1-second bins touched within the bar over
    # the bar's total seconds. ``dt.floor`` is timezone-safe and avoids the
    # int64 cast that silently mishandles tz-aware datetimes.
    bar_seconds = int(pd.Timedelta(bar_size).total_seconds())
    if bar_seconds > 0:
        sub["second_floor"] = sub["timestamp"].dt.floor("1s")
        unique_seconds = sub.groupby("bar_open_time")["second_floor"].nunique()
        out["coverage_ratio"] = (unique_seconds.reindex(out.index, fill_value=0) / bar_seconds).clip(upper=1.0)
    else:
        out["coverage_ratio"] = 0.0
    out["coverage_ratio"] = out["coverage_ratio"].astype(float)
    out["source_quality"] = np.where(out["trade_count"] > 0, "complete", "empty")

    out = out.reset_index().rename(columns={"index": "bar_open_time"})
    out.insert(0, "symbol", symbol)
    out.insert(2, "bar_size", bar_size)
    return out[
        [
            "symbol", "bar_open_time", "bar_size",
            "trade_count", "volume", "turnover",
            "buy_volume", "sell_volume", "buy_turnover", "sell_turnover",
            "taker_buy_ratio", "buy_sell_imbalance",
            "cvd_delta_volume", "cvd_delta_turnover",
            "large_trade_threshold",
            "large_buy_count", "large_sell_count",
            "large_buy_turnover", "large_sell_turnover",
            "coverage_ratio", "source_quality",
        ]
    ]


def _shard_path(continuous_root: Path, symbol: str, day: date, bar_size: str) -> Path:
    """Per-(symbol, bar_size, month) parquet shard path."""
    yyyymm = f"{day:%Y-%m}"
    return continuous_root / symbol / bar_size / f"{yyyymm}.parquet"


def _append_shard(features: pd.DataFrame, shard_path: Path) -> int:
    """Merge ``features`` into the shard at ``shard_path`` (de-dupe by
    bar_open_time). Returns the number of rows actually written
    (added or replaced)."""
    ensure_dir(shard_path.parent)
    if features.empty:
        return 0
    if shard_path.exists():
        existing = pd.read_parquet(shard_path)
        combined = pd.concat([existing, features], ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "bar_open_time", "bar_size"], keep="last")
        combined = combined.sort_values(["bar_size", "bar_open_time"]).reset_index(drop=True)
    else:
        combined = features.sort_values(["bar_size", "bar_open_time"]).reset_index(drop=True)
    combined.to_parquet(shard_path, index=False)
    return int(len(features))


# --------------------------------------------------------------------------------------
# Driver — (symbol, day) outcomes + per-symbol day list orchestration
# --------------------------------------------------------------------------------------


def _ensure_aggtrades_zip(
    symbol: str,
    day: date,
    cfg: ContinuousCvdConfig,
) -> tuple[Path, bool, str]:
    """Return (path, is_present, status). When the zip is absent and
    ``download_if_missing`` is True, attempt one download; otherwise
    return ``status='missing'`` so the driver records it in the
    coverage report."""
    zip_path = aggtrades_zip_path(cfg.history_root, symbol, day)
    if zip_path.exists():
        return zip_path, True, "cached"
    missing_marker = zip_path.with_suffix(zip_path.suffix + ".missing")
    if missing_marker.exists():
        return zip_path, False, "marker_missing"
    if not cfg.download_if_missing:
        return zip_path, False, "not_downloaded"
    download_aggtrades_day(symbol, day, cfg.orderflow_cfg)
    if zip_path.exists():
        return zip_path, True, "downloaded"
    return zip_path, False, "download_failed"


def backfill_symbol_day(
    symbol: str,
    day: date,
    cfg: ContinuousCvdConfig | None = None,
) -> DayOutcome:
    """Process one (symbol, day) end-to-end."""
    cfg = cfg or ContinuousCvdConfig()
    zip_path, present, status = _ensure_aggtrades_zip(symbol, day, cfg)
    if not present:
        return DayOutcome(symbol=symbol, day=day, source_quality="missing", trade_count=0, coverage_seconds=0, error=status)
    try:
        trades_raw = load_aggtrades_zip(zip_path)
    except Exception as exc:  # corrupted zip etc.
        return DayOutcome(symbol=symbol, day=day, source_quality="error", trade_count=0, coverage_seconds=0, error=str(exc))
    if trades_raw.empty:
        return DayOutcome(symbol=symbol, day=day, source_quality="empty", trade_count=0, coverage_seconds=0)
    trades_raw = trades_raw.copy()
    trades_raw["symbol"] = symbol  # normalize sometimes leaves it blank
    bars_total = 0
    coverage_seconds = 0
    for bar_size in cfg.bar_sizes:
        features = compute_continuous_features(
            trades_raw, symbol=symbol, day=day, bar_size=bar_size, large_quantile=cfg.large_trade_turnover_quantile
        )
        shard_path = _shard_path(cfg.continuous_root, symbol, day, bar_size)
        bars_total += _append_shard(features, shard_path)
        if bar_size == "1min":
            coverage_seconds = int((features["coverage_ratio"] * 60).sum())
    return DayOutcome(
        symbol=symbol,
        day=day,
        source_quality="complete" if bars_total > 0 else "empty",
        trade_count=int(len(trades_raw)),
        coverage_seconds=coverage_seconds,
    )


def backfill_symbol_days(
    symbols: list[str],
    start_day: date,
    end_day: date,
    cfg: ContinuousCvdConfig | None = None,
    *,
    progress_every: int = 25,
) -> BackfillStats:
    """Backfill every (symbol, day) in the inclusive [start_day, end_day] range."""
    cfg = cfg or ContinuousCvdConfig()
    ensure_dir(cfg.continuous_root)
    stats = BackfillStats()
    cur = start_day
    days = []
    while cur <= end_day:
        days.append(cur)
        cur = cur + timedelta(days=1)
    total_tasks = len(symbols) * len(days)
    print(f"vData continuous CVD: {len(symbols)} symbols × {len(days)} days = {total_tasks} tasks", flush=True)
    idx = 0
    for symbol in symbols:
        for day in days:
            idx += 1
            outcome = backfill_symbol_day(symbol, day, cfg)
            stats.outcomes.append(outcome)
            stats.days_attempted += 1
            if outcome.source_quality == "complete":
                stats.days_complete += 1
            elif outcome.source_quality == "empty":
                stats.days_empty += 1
            elif outcome.source_quality == "missing":
                stats.days_missing += 1
            else:
                stats.days_error += 1
            if outcome.trade_count > 0:
                stats.bars_written += outcome.trade_count  # rough bars proxy
            if idx % progress_every == 0:
                print(
                    f"  {idx}/{total_tasks} done — complete={stats.days_complete} "
                    f"empty={stats.days_empty} missing={stats.days_missing} error={stats.days_error}",
                    flush=True,
                )
    return stats


def write_coverage_report(stats: BackfillStats, cfg: ContinuousCvdConfig | None = None) -> Path:
    cfg = cfg or ContinuousCvdConfig()
    ensure_dir(cfg.history_root)
    df = pd.DataFrame([{
        "symbol": o.symbol,
        "day": pd.Timestamp(o.day),
        "source_quality": o.source_quality,
        "trade_count": o.trade_count,
        "coverage_seconds": o.coverage_seconds,
        "error": o.error,
    } for o in stats.outcomes])
    if COVERAGE_REPORT_PATH.exists() and not df.empty:
        prev = pd.read_parquet(COVERAGE_REPORT_PATH)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset=["symbol", "day"], keep="last"
        )
    df.to_parquet(COVERAGE_REPORT_PATH, index=False)
    return COVERAGE_REPORT_PATH


def write_quality_audit(symbols: list[str], cfg: ContinuousCvdConfig | None = None) -> Path:
    """Walk the continuous shards for each (symbol, bar_size), summarise
    NaN counts / zero-trade bars / mean coverage_ratio.

    Useful as a one-shot data-quality check after a backfill batch."""
    cfg = cfg or ContinuousCvdConfig()
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for bar_size in cfg.bar_sizes:
            sym_dir = cfg.continuous_root / symbol / bar_size
            if not sym_dir.exists():
                continue
            shards = sorted(sym_dir.glob("*.parquet"))
            for shard in shards:
                df = pd.read_parquet(shard)
                rows.append({
                    "symbol": symbol,
                    "bar_size": bar_size,
                    "shard": shard.name,
                    "n_bars": int(len(df)),
                    "n_zero_trade_bars": int((df["trade_count"] == 0).sum()),
                    "mean_coverage_ratio": float(df["coverage_ratio"].mean()),
                    "n_nan_imbalance": int(df["buy_sell_imbalance"].isna().sum()),
                    "min_taker_buy_ratio": float(df["taker_buy_ratio"].min()),
                    "max_taker_buy_ratio": float(df["taker_buy_ratio"].max()),
                })
    audit = pd.DataFrame(rows)
    audit.to_parquet(QUALITY_AUDIT_PATH, index=False)
    return QUALITY_AUDIT_PATH


__all__ = [
    "ContinuousCvdConfig",
    "DayOutcome",
    "BackfillStats",
    "CONTINUOUS_ROOT",
    "COVERAGE_REPORT_PATH",
    "QUALITY_AUDIT_PATH",
    "DEFAULT_BAR_SIZES",
    "compute_continuous_features",
    "backfill_symbol_day",
    "backfill_symbol_days",
    "write_coverage_report",
    "write_quality_audit",
]
