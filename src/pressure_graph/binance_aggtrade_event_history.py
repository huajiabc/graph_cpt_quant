"""Incremental Binance USD-M aggTrade windows for frozen research events."""
from __future__ import annotations

import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.orderflow_history import (
    AGG_TRADE_COLUMNS,
    OrderflowHistoryConfig,
    download_aggtrades_day,
    normalize_aggtrades,
)


BASE_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
OUTPUT_ROOT = Path("data/external/binance_um_aggtrade_event_windows")
FEATURE_PATH = OUTPUT_ROOT / "event_window_features.parquet"
MANIFEST_PATH = OUTPUT_ROOT / "last_collection.csv"


@dataclass(frozen=True)
class AggTradeEventConfig:
    output_root: Path = OUTPUT_ROOT
    timeout_seconds: int = 30
    retry_attempts: int = 5
    retry_sleep_seconds: float = 1.0
    request_interval_seconds: float = 0.65
    page_limit: int = 1_000
    max_pages: int = 100
    flush_every: int = 10


@dataclass(frozen=True)
class AggTradeArchiveEventConfig:
    output_root: Path = OUTPUT_ROOT
    cache_root: Path = OUTPUT_ROOT / "archive_cache"
    timeout_seconds: int = 180
    flush_every_archives: int = 5
    delete_archives_after_success: bool = True

    @property
    def orderflow_config(self) -> OrderflowHistoryConfig:
        return OrderflowHistoryConfig(
            history_root=self.cache_root,
            timeout_seconds=self.timeout_seconds,
            download_workers=1,
        )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["agg_trade_id", "price", "quantity", "timestamp", "buyer_maker"]
    )


def normalize_rest_aggtrades(payload: list[dict[str, object]]) -> pd.DataFrame:
    if not payload:
        return _empty_trades()
    frame = pd.DataFrame(payload)
    required = {"a", "p", "q", "T", "m"}
    if not required.issubset(frame.columns):
        return _empty_trades()
    output = pd.DataFrame(
        {
            "agg_trade_id": pd.to_numeric(frame["a"], errors="coerce"),
            "price": pd.to_numeric(frame["p"], errors="coerce"),
            "quantity": pd.to_numeric(frame["q"], errors="coerce"),
            "timestamp": pd.to_datetime(
                pd.to_numeric(frame["T"], errors="coerce"),
                unit="ms",
                utc=True,
                errors="coerce",
            ),
            "buyer_maker": frame["m"].astype(bool),
        }
    )
    return (
        output.dropna(subset=["agg_trade_id", "price", "quantity", "timestamp"])
        .drop_duplicates("agg_trade_id", keep="last")
        .sort_values(["timestamp", "agg_trade_id"])
        .reset_index(drop=True)
    )


def fetch_aggtrade_window(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client,
    cfg: AggTradeEventConfig = AggTradeEventConfig(),
) -> tuple[pd.DataFrame, int, int]:
    """Fetch [start, end) with id pagination and return trades/pages/weight."""
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    pages: list[pd.DataFrame] = []
    next_id: int | None = None
    maximum_weight = 0
    for page in range(cfg.max_pages):
        params: dict[str, object] = {"symbol": symbol, "limit": cfg.page_limit}
        if next_id is None:
            params.update(
                {
                    "startTime": int(start.timestamp() * 1_000),
                    "endTime": int(end.timestamp() * 1_000) - 1,
                }
            )
        else:
            params["fromId"] = next_id
        response: httpx.Response | None = None
        for attempt in range(cfg.retry_attempts):
            response = client.get(BASE_URL, params=params)
            used = int(response.headers.get("x-mbx-used-weight-1m", "0") or 0)
            maximum_weight = max(maximum_weight, used)
            if response.status_code == 200:
                break
            if response.status_code not in {418, 429, 500, 502, 503, 504}:
                response.raise_for_status()
            retry_after = float(response.headers.get("retry-after", "0") or 0)
            wait = max(
                cfg.retry_sleep_seconds * (2**attempt),
                min(retry_after, 30.0),
            )
            time.sleep(wait)
        if response is None or response.status_code != 200:
            raise RuntimeError(f"aggTrades request failed for {symbol} {start}")
        payload = response.json()
        local = normalize_rest_aggtrades(payload)
        if local.empty:
            break
        pages.append(local)
        last_id = int(local["agg_trade_id"].max())
        next_id = last_id + 1
        if len(local) < cfg.page_limit or local["timestamp"].max() >= end:
            break
        time.sleep(cfg.request_interval_seconds)
        if page == cfg.max_pages - 1:
            raise RuntimeError(f"aggTrades pagination limit reached for {symbol} {start}")
    if not pages:
        return _empty_trades(), 0, maximum_weight
    trades = (
        pd.concat(pages, ignore_index=True)
        .drop_duplicates("agg_trade_id", keep="last")
        .sort_values(["timestamp", "agg_trade_id"])
    )
    trades = trades[
        trades["timestamp"].ge(start) & trades["timestamp"].lt(end)
    ].reset_index(drop=True)
    return trades, len(pages), maximum_weight


def summarize_aggtrade_window(
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source_sign: float,
) -> dict[str, object]:
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    local = trades[
        trades["timestamp"].ge(start) & trades["timestamp"].lt(end)
    ].copy()
    if local.empty:
        return {
            "trade_count": 0,
            "first_trade_time": pd.NaT,
            "last_trade_time": pd.NaT,
            "covered_seconds": 0,
            "price_return": np.nan,
            "aligned_price_return": np.nan,
            "buy_turnover": 0.0,
            "sell_turnover": 0.0,
            "buy_sell_imbalance": np.nan,
            "early_aligned_imbalance": np.nan,
            "middle_aligned_imbalance": np.nan,
            "late_aligned_imbalance": np.nan,
            "aligned_flow_exhaustion": np.nan,
            "late_flow_opposes_source": False,
            "large_aligned_imbalance": np.nan,
        }
    if "turnover" not in local.columns:
        size_column = "quantity" if "quantity" in local.columns else "size"
        local["turnover"] = local["price"] * local[size_column]
    if "buyer_maker" in local.columns:
        local["is_buy"] = ~local["buyer_maker"]
    else:
        local["is_buy"] = local["side"].astype(str).str.lower().eq("buy")
    split1 = start + (end - start) / 3
    split2 = start + 2 * (end - start) / 3
    local["third"] = np.select(
        [local["timestamp"].lt(split1), local["timestamp"].lt(split2)],
        ["early", "middle"],
        default="late",
    )

    def imbalance(frame: pd.DataFrame) -> float:
        total = float(frame["turnover"].sum())
        if total <= 0:
            return math.nan
        buy = float(frame.loc[frame["is_buy"], "turnover"].sum())
        sell = total - buy
        return (buy - sell) / total

    total_imbalance = imbalance(local)
    thirds = {
        name: imbalance(local[local["third"].eq(name)])
        for name in ("early", "middle", "late")
    }
    large_threshold = float(local["turnover"].quantile(0.95))
    large = local[local["turnover"].ge(large_threshold)]
    aligned_thirds = {
        name: float(source_sign * value) if np.isfinite(value) else math.nan
        for name, value in thirds.items()
    }
    price_return = float(local["price"].iloc[-1] / local["price"].iloc[0] - 1.0)
    seconds = local["timestamp"].dt.floor("1s").nunique()
    return {
        "trade_count": len(local),
        "first_trade_time": local["timestamp"].min(),
        "last_trade_time": local["timestamp"].max(),
        "covered_seconds": int(seconds),
        "price_return": price_return,
        "aligned_price_return": float(source_sign * price_return),
        "buy_turnover": float(local.loc[local["is_buy"], "turnover"].sum()),
        "sell_turnover": float(local.loc[~local["is_buy"], "turnover"].sum()),
        "buy_sell_imbalance": total_imbalance,
        "aligned_buy_sell_imbalance": float(source_sign * total_imbalance),
        "early_aligned_imbalance": aligned_thirds["early"],
        "middle_aligned_imbalance": aligned_thirds["middle"],
        "late_aligned_imbalance": aligned_thirds["late"],
        "aligned_flow_exhaustion": (
            aligned_thirds["early"] - aligned_thirds["late"]
        ),
        "late_flow_opposes_source": bool(aligned_thirds["late"] < 0),
        "large_aligned_imbalance": float(source_sign * imbalance(large)),
    }


def load_aggtrades_zip_windows(
    path: Path,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Stream an archive and retain only rows inside the requested time windows."""
    if not windows:
        return pd.DataFrame(
            columns=[
                "exchange",
                "symbol",
                "timestamp",
                "execId",
                "price",
                "size",
                "turnover",
                "side",
            ]
        )
    normalized_windows = [
        (pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC"))
        for start, end in windows
    ]
    overall_start = min(start for start, _ in normalized_windows)
    overall_end = max(end for _, end in normalized_windows)
    retained: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with archive.open(names[0]) as handle:
            chunks = pd.read_csv(
                handle,
                header=None,
                names=AGG_TRADE_COLUMNS,
                usecols=range(len(AGG_TRADE_COLUMNS)),
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in chunks:
                normalized = normalize_aggtrades(chunk)
                if normalized.empty:
                    continue
                normalized = normalized[
                    normalized["timestamp"].ge(overall_start)
                    & normalized["timestamp"].lt(overall_end)
                ]
                if normalized.empty:
                    continue
                mask = pd.Series(False, index=normalized.index)
                for start, end in normalized_windows:
                    mask |= normalized["timestamp"].ge(start) & normalized[
                        "timestamp"
                    ].lt(end)
                if mask.any():
                    retained.append(normalized[mask])
    if not retained:
        return pd.DataFrame()
    return (
        pd.concat(retained, ignore_index=True)
        .drop_duplicates("execId", keep="last")
        .sort_values(["timestamp", "execId"])
        .reset_index(drop=True)
    )


def build_extreme_overshoot_tasks(feature_events: pd.DataFrame) -> pd.DataFrame:
    selected = feature_events[
        feature_events["source_scope"].eq("COMMUNITY_COHERENT_INDEX_SHOCK")
        & feature_events["family"].eq("TRADE_VS_MARK_OVERSHOOT_FADE")
        & feature_events["source_setting"].eq("z2.0")
        & feature_events["receiver_z_threshold"].eq(2.0)
        & feature_events["receiver_count"].ge(3)
    ].copy()
    rows: list[dict[str, object]] = []
    for item in selected.itertuples(index=False):
        feature_time = pd.Timestamp(item.feature_time)
        if feature_time.tzinfo is None:
            feature_time = feature_time.tz_localize("UTC")
        else:
            feature_time = feature_time.tz_convert("UTC")
        for symbol in [name for name in str(item.receivers).split("|") if name]:
            rows.append(
                {
                    "task_id": f"{item.source_event_id}|{symbol}",
                    "source_event_id": item.source_event_id,
                    "feature_time": feature_time,
                    "window_start": feature_time - pd.Timedelta(minutes=15),
                    "window_end": feature_time,
                    "period": item.period,
                    "community_id": item.community_id,
                    "source_sign": float(item.source_sign),
                    "symbol": symbol,
                }
            )
    return pd.DataFrame(rows).drop_duplicates("task_id").sort_values(
        ["feature_time", "community_id", "symbol"]
    ).reset_index(drop=True)


def _merge_features(path: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    frames = [pd.DataFrame(rows)] if rows else []
    if path.exists():
        frames.insert(0, pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("task_id", keep="last")
        .sort_values(["feature_time", "community_id", "symbol"])
        .reset_index(drop=True)
    )
    merged.to_parquet(path, index=False)
    return merged


def collect_aggtrade_event_windows(
    tasks: pd.DataFrame,
    cfg: AggTradeEventConfig = AggTradeEventConfig(),
    max_tasks: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ensure_dir(cfg.output_root)
    feature_path = root / FEATURE_PATH.name
    existing = pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame()
    completed = set(existing.get("task_id", pd.Series(dtype=str)).astype(str))
    pending = tasks[~tasks["task_id"].astype(str).isin(completed)].copy()
    if max_tasks is not None:
        pending = pending.head(max_tasks)
    buffered: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-event-orderflow/1.0"},
    ) as client:
        for index, item in enumerate(pending.itertuples(index=False), start=1):
            started = time.monotonic()
            try:
                trades, pages, weight = fetch_aggtrade_window(
                    str(item.symbol),
                    pd.Timestamp(item.window_start),
                    pd.Timestamp(item.window_end),
                    client,
                    cfg,
                )
                features = summarize_aggtrade_window(
                    trades,
                    pd.Timestamp(item.window_start),
                    pd.Timestamp(item.window_end),
                    float(item.source_sign),
                )
                buffered.append(
                    {
                        **item._asdict(),
                        **features,
                        "pages": pages,
                        "maximum_reported_weight_1m": weight,
                    }
                )
                status = "complete" if len(trades) else "empty"
                error = ""
            except Exception as exc:
                pages = 0
                weight = 0
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            manifest_rows.append(
                {
                    "task_id": item.task_id,
                    "symbol": item.symbol,
                    "feature_time": item.feature_time,
                    "status": status,
                    "pages": pages,
                    "maximum_reported_weight_1m": weight,
                    "elapsed_seconds": time.monotonic() - started,
                    "error": error,
                }
            )
            if len(buffered) >= cfg.flush_every:
                existing = _merge_features(feature_path, buffered)
                buffered = []
            print(
                f"aggTrades {index}/{len(pending)} {item.symbol} "
                f"{item.feature_time} {status} pages={pages}",
                flush=True,
            )
            time.sleep(cfg.request_interval_seconds)
    if buffered:
        existing = _merge_features(feature_path, buffered)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(root / MANIFEST_PATH.name, index=False)
    return existing, manifest


def collect_aggtrade_event_archives(
    tasks: pd.DataFrame,
    cfg: AggTradeArchiveEventConfig = AggTradeArchiveEventConfig(),
    max_archives: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect event windows from temporary daily archives, deleting each after use."""
    root = ensure_dir(cfg.output_root)
    feature_path = root / FEATURE_PATH.name
    existing = pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame()
    completed = set(existing.get("task_id", pd.Series(dtype=str)).astype(str))
    pending = tasks[~tasks["task_id"].astype(str).isin(completed)].copy()
    pending["archive_day"] = pd.to_datetime(
        pending["window_start"], utc=True
    ).dt.date
    groups = list(pending.groupby(["symbol", "archive_day"], sort=True))
    if max_archives is not None:
        groups = groups[:max_archives]
    buffered: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    orderflow_cfg = cfg.orderflow_config
    for index, ((symbol, archive_day), local_tasks) in enumerate(groups, start=1):
        started = time.monotonic()
        status = "error"
        error = ""
        archive_bytes = 0
        try:
            downloaded = download_aggtrades_day(
                str(symbol), archive_day, orderflow_cfg
            )
            if downloaded is None:
                raise FileNotFoundError(f"archive not found: {symbol} {archive_day}")
            archive_bytes = downloaded.stat().st_size
            try:
                windows = [
                    (pd.Timestamp(item.window_start), pd.Timestamp(item.window_end))
                    for item in local_tasks.itertuples(index=False)
                ]
                trades = load_aggtrades_zip_windows(downloaded, windows)
            except Exception:
                downloaded.unlink(missing_ok=True)
                downloaded = download_aggtrades_day(
                    str(symbol), archive_day, orderflow_cfg
                )
                if downloaded is None:
                    raise
                archive_bytes = downloaded.stat().st_size
                trades = load_aggtrades_zip_windows(downloaded, windows)
            if trades.empty:
                raise ValueError(f"empty archive: {symbol} {archive_day}")
            for item in local_tasks.itertuples(index=False):
                features = summarize_aggtrade_window(
                    trades,
                    pd.Timestamp(item.window_start),
                    pd.Timestamp(item.window_end),
                    float(item.source_sign),
                )
                buffered.append(
                    {
                        **{
                            key: value
                            for key, value in item._asdict().items()
                            if key != "archive_day"
                        },
                        **features,
                        "pages": np.nan,
                        "maximum_reported_weight_1m": np.nan,
                        "source_mode": "checked_zip_crc_archive",
                        "source_archive": downloaded.name,
                        "source_archive_bytes": archive_bytes,
                    }
                )
            status = "complete"
            if cfg.delete_archives_after_success:
                downloaded.unlink(missing_ok=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        manifest_rows.append(
            {
                "symbol": symbol,
                "archive_day": archive_day,
                "event_windows": len(local_tasks),
                "status": status,
                "archive_bytes": archive_bytes,
                "elapsed_seconds": time.monotonic() - started,
                "error": error,
            }
        )
        if index % cfg.flush_every_archives == 0 and buffered:
            existing = _merge_features(feature_path, buffered)
            buffered = []
        print(
            f"aggTrades archive {index}/{len(groups)} {symbol} {archive_day} "
            f"windows={len(local_tasks)} {status} bytes={archive_bytes}",
            flush=True,
        )
    if buffered:
        existing = _merge_features(feature_path, buffered)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(root / "last_archive_collection.csv", index=False)
    return existing, manifest


__all__ = [
    "AggTradeEventConfig",
    "AggTradeArchiveEventConfig",
    "build_extreme_overshoot_tasks",
    "collect_aggtrade_event_archives",
    "collect_aggtrade_event_windows",
    "fetch_aggtrade_window",
    "load_aggtrades_zip_windows",
    "normalize_rest_aggtrades",
    "summarize_aggtrade_window",
]
