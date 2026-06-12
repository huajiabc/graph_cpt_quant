"""Historical event-window orderflow backfill from Binance USDT-M aggTrades archives.

The v0.8 live shadow only accrues taker-flow context for paper trades going
forward, which leaves the selected-vs-skipped ranking question underpowered for
months. This module rebuilds the same v0.8 window features (taker buy ratio,
CVD delta, large-trade pressure) for the FULL historical CIC candidate stream
(v0.9D capacity trade cache) using Binance public aggTrades daily archives as
the tape source. ``binance_proxy`` is an accepted source preference in the
v0.8.1 demand queue; Bybit-native archives stay blocked from the research box.

Feature semantics intentionally reuse ``pressure_graph.orderflow.summarize_trade_window``
so historical and live-shadow columns stay directly comparable.

Window contract (15m bars, signal_time = shock-bar close, entry at bar open):

    shock_bar        [signal_time - 15m, signal_time)
    pullback_window  [signal_time, entry_time - 15m)        pre-entry
    reclaim_bar      [entry_time - 15m, entry_time)         pre-entry
    pre_entry_all    [signal_time - 15m, entry_time)        pre-entry
    entry_bar        [entry_time, entry_time + 15m)         DIAGNOSTIC ONLY
    post_entry_1h    [entry_time, entry_time + 60m)         DIAGNOSTIC ONLY

Ranking experiments must only consume pre-entry windows; entry/post windows
exist for follow-through diagnostics and would leak future information into a
capacity decision made at entry_time.
"""
from __future__ import annotations

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, read_parquet, write_parquet
from pressure_graph.orderflow import summarize_trade_window

BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data"
HISTORY_ROOT = Path("data/orderflow_history/binance_um")
REPORT_ROOT = Path("reports/v1_1_orderflow_burst_ranking")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")

DEFAULT_CANDIDATES = ("CIC1_beta_extreme", "CIC2_beta_broad", "MIR1_reference")

# Bybit linear -> Binance USDT-M futures aliases that the generic variant
# generator cannot derive (multiplier naming differs between venues).
BYBIT_TO_BINANCE_UM_ALIASES = {
    "SHIB1000USDT": "1000SHIBUSDT",
    "PEPE1000USDT": "1000PEPEUSDT",
    "FLOKI1000USDT": "1000FLOKIUSDT",
    "BONK1000USDT": "1000BONKUSDT",
}

AGG_TRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]

PRE_ENTRY_WINDOWS = ("shock_bar", "pullback_window", "reclaim_bar", "pre_entry_all")
DIAGNOSTIC_WINDOWS = ("entry_bar", "post_entry_1h")
EVENT_WINDOWS = PRE_ENTRY_WINDOWS + DIAGNOSTIC_WINDOWS

BAR = pd.Timedelta(minutes=15)
EVENT_PAD_BEFORE = pd.Timedelta(minutes=30)
EVENT_PAD_AFTER = pd.Timedelta(minutes=90)


@dataclass(frozen=True)
class OrderflowHistoryConfig:
    history_root: Path = HISTORY_ROOT
    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES
    download_workers: int = 8
    timeout_seconds: int = 120
    large_trade_quantile: float = 0.95
    keep_zips: bool = True
    max_events: int | None = None
    extra_aliases: dict[str, str] = field(default_factory=dict)


def aggtrades_daily_url(binance_symbol: str, day: date) -> str:
    stamp = day.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/daily/aggTrades/"
        f"{binance_symbol}/{binance_symbol}-aggTrades-{stamp}.zip"
    )


def aggtrades_zip_path(history_root: Path, binance_symbol: str, day: date) -> Path:
    stamp = day.strftime("%Y-%m-%d")
    return history_root / "raw" / "aggTrades" / binance_symbol / f"{binance_symbol}-aggTrades-{stamp}.zip"


def binance_symbol_candidates(bybit_symbol: str, extra_aliases: dict[str, str] | None = None) -> list[str]:
    """Ordered Binance symbol guesses for one Bybit linear symbol."""
    symbol = bybit_symbol.upper().strip()
    candidates: list[str] = []
    for alias_map in (extra_aliases or {}, BYBIT_TO_BINANCE_UM_ALIASES):
        alias = alias_map.get(symbol)
        if alias:
            candidates.append(alias)
    candidates.append(symbol)
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    core = base
    for prefix in ("1000000", "100000", "10000", "1000"):
        if core.startswith(prefix) and len(core) > len(prefix):
            core = core[len(prefix):]
            break
    for suffix in ("1000000", "10000", "1000"):
        if core.endswith(suffix) and len(core) > len(suffix):
            core = core[: -len(suffix)]
            break
    for form in (f"{core}USDT", f"1000{core}USDT", f"1000000{core}USDT"):
        candidates.append(form)
    return list(dict.fromkeys(candidates))


def download_aggtrades_day(
    binance_symbol: str,
    day: date,
    cfg: OrderflowHistoryConfig,
) -> Path | None:
    """Download one daily zip. Returns the cached path, or None on 404."""
    path = aggtrades_zip_path(cfg.history_root, binance_symbol, day)
    if path.exists() and path.stat().st_size > 0:
        return path
    miss_marker = path.with_suffix(".zip.missing")
    if miss_marker.exists():
        return None
    ensure_dir(path.parent)
    request = Request(
        aggtrades_daily_url(binance_symbol, day),
        headers={"User-Agent": "pressure-graph/0.1 orderflow-history"},
    )
    try:
        with urlopen(request, timeout=cfg.timeout_seconds) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            miss_marker.write_text("404", encoding="utf-8")
            return None
        raise
    tmp_path = path.with_suffix(".zip.part")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)
    return path


def load_aggtrades_zip(path: Path) -> pd.DataFrame:
    """Read one daily zip into the v0.8 normalized trade schema."""
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.endswith(".csv")]
        if not names:
            return _empty_normalized()
        with zf.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=None, low_memory=False)
    if len(frame.columns) >= len(AGG_TRADE_COLUMNS):
        frame = frame.iloc[:, : len(AGG_TRADE_COLUMNS)]
        frame.columns = AGG_TRADE_COLUMNS
    else:
        return _empty_normalized()
    return normalize_aggtrades(frame)


def normalize_aggtrades(frame: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """aggTrades rows -> normalized trade frame (header rows coerce to NaN and drop)."""
    if frame.empty:
        return _empty_normalized()
    out = pd.DataFrame()
    price = pd.to_numeric(frame["price"], errors="coerce")
    size = pd.to_numeric(frame["quantity"], errors="coerce")
    transact = pd.to_numeric(frame["transact_time"], errors="coerce")
    # Binance archives switched some markets from ms to us epochs.
    median = float(transact.dropna().median()) if transact.notna().any() else 0.0
    unit = "us" if median > 1e14 else "ms"
    maker = frame["is_buyer_maker"].map(_parse_bool)
    out["timestamp"] = pd.to_datetime(transact, unit=unit, utc=True, errors="coerce")
    out["price"] = price
    out["size"] = size
    out["turnover"] = price * size
    out["side"] = np.where(maker.fillna(False).astype(bool), "sell", "buy")
    out["execId"] = frame["agg_trade_id"].astype(str)
    out["exchange"] = "binance"
    out["symbol"] = symbol
    out = out.dropna(subset=["timestamp", "price", "size"])
    return (
        out[["exchange", "symbol", "timestamp", "execId", "price", "size", "turnover", "side"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "t"}


def _empty_normalized() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["exchange", "symbol", "timestamp", "execId", "price", "size", "turnover", "side"]
    )


def event_windows(signal_time: pd.Timestamp, entry_time: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Window map for one candidate event. Empty pullback collapses to zero span."""
    signal_time = _utc(signal_time)
    entry_time = _utc(entry_time)
    pullback_end = max(signal_time, entry_time - BAR)
    return {
        "shock_bar": (signal_time - BAR, signal_time),
        "pullback_window": (signal_time, pullback_end),
        "reclaim_bar": (entry_time - BAR, entry_time),
        "pre_entry_all": (signal_time - BAR, entry_time),
        "entry_bar": (entry_time, entry_time + BAR),
        "post_entry_1h": (entry_time, entry_time + pd.Timedelta(hours=1)),
    }


def _utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def signal_event_id(exchange: str, symbol: str, signal_time: pd.Timestamp, candidate: str) -> str:
    """Mirror reports.v07d1._signal_id for a single event."""
    return f"{exchange}|{symbol}|{_utc(signal_time)}|{candidate}"


def extract_events(trades: pd.DataFrame, candidates: tuple[str, ...], max_events: int | None = None) -> pd.DataFrame:
    """Unique candidate events from the capacity trade cache (cost frames deduped)."""
    if trades.empty:
        return pd.DataFrame(
            columns=["exchange", "symbol", "candidate", "signal_time", "entry_time", "signal_id"]
        )
    out = trades.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce")
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out = out[out["candidate"].astype(str).isin(candidates)]
    out = out.dropna(subset=["signal_time", "entry_time"])
    out = (
        out[["exchange", "symbol", "candidate", "signal_time", "entry_time"]]
        .drop_duplicates()
        .sort_values(["signal_time", "symbol", "candidate"])
        .reset_index(drop=True)
    )
    out["signal_id"] = [
        signal_event_id(row.exchange, row.symbol, row.signal_time, row.candidate)
        for row in out.itertuples(index=False)
    ]
    if max_events is not None:
        out = out.head(max_events).copy()
    return out


def _event_day_span(event: pd.Series) -> list[date]:
    start = (_utc(event["signal_time"]) - EVENT_PAD_BEFORE).date()
    end = (_utc(event["entry_time"]) + EVENT_PAD_AFTER).date()
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def resolve_binance_symbol(
    bybit_symbol: str,
    probe_days: list[date],
    cfg: OrderflowHistoryConfig,
    symbol_map: dict[str, str],
) -> str | None:
    """Resolve and cache the Binance UM symbol for a Bybit symbol via archive probes."""
    cached = symbol_map.get(bybit_symbol)
    if cached is not None:
        return cached or None
    for candidate in binance_symbol_candidates(bybit_symbol, cfg.extra_aliases):
        for day in probe_days:
            try:
                path = download_aggtrades_day(candidate, day, cfg)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"[orderflow-history] probe failed {candidate} {day}: {exc}", flush=True)
                continue
            if path is not None:
                symbol_map[bybit_symbol] = candidate
                return candidate
    symbol_map[bybit_symbol] = ""
    return None


def _symbol_map_path(cfg: OrderflowHistoryConfig) -> Path:
    return cfg.history_root / "symbol_map.json"


def _load_symbol_map(cfg: OrderflowHistoryConfig) -> dict[str, str]:
    path = _symbol_map_path(cfg)
    if not path.exists():
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_symbol_map(symbol_map: dict[str, str], cfg: OrderflowHistoryConfig) -> None:
    path = _symbol_map_path(cfg)
    ensure_dir(path.parent)
    path.write_text(json.dumps(dict(sorted(symbol_map.items())), indent=2), encoding="utf-8")


def _download_symbol_days(
    binance_symbol: str,
    days: list[date],
    cfg: OrderflowHistoryConfig,
) -> dict[date, Path | None]:
    results: dict[date, Path | None] = {}
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
        futures = {
            day: pool.submit(download_aggtrades_day, binance_symbol, day, cfg) for day in days
        }
        for day, future in futures.items():
            try:
                results[day] = future.result()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"[orderflow-history] download failed {binance_symbol} {day}: {exc}", flush=True)
                results[day] = None
    return results


def build_event_orderflow(
    events: pd.DataFrame,
    cfg: OrderflowHistoryConfig,
) -> pd.DataFrame:
    """Compute v0.8-style window features for each event, grouped per symbol."""
    if events.empty:
        return pd.DataFrame()
    ensure_dir(cfg.history_root)
    symbol_map = _load_symbol_map(cfg)
    rows: list[dict[str, object]] = []
    grouped = events.groupby("symbol", sort=True)
    for symbol_idx, (symbol, group) in enumerate(grouped, start=1):
        day_set: set[date] = set()
        for _, event in group.iterrows():
            day_set.update(_event_day_span(event))
        days = sorted(day_set)
        probe_days = [days[len(days) // 2], days[0], days[-1]]
        binance_symbol = resolve_binance_symbol(str(symbol), probe_days, cfg, symbol_map)
        if binance_symbol is None:
            for _, event in group.iterrows():
                rows.append(_unmapped_row(event))
            print(
                f"[orderflow-history] {symbol_idx}/{len(grouped)} {symbol}: no Binance UM listing",
                flush=True,
            )
            continue
        day_paths = _download_symbol_days(binance_symbol, days, cfg)
        day_frames: dict[date, pd.DataFrame] = {}
        for day, path in day_paths.items():
            if path is None:
                continue
            frame = load_aggtrades_zip(path)
            if not frame.empty:
                day_frames[day] = frame
        for _, event in group.iterrows():
            rows.append(_event_row(event, binance_symbol, day_frames, cfg))
        covered = sum(1 for day in days if day in day_frames)
        print(
            f"[orderflow-history] {symbol_idx}/{len(grouped)} {symbol} -> {binance_symbol}: "
            f"events={len(group)} days={len(days)} days_with_tape={covered}",
            flush=True,
        )
        _save_symbol_map(symbol_map, cfg)
    _save_symbol_map(symbol_map, cfg)
    return pd.DataFrame(rows)


def _unmapped_row(event: pd.Series) -> dict[str, object]:
    row = {
        "signal_id": event["signal_id"],
        "exchange": event["exchange"],
        "symbol": event["symbol"],
        "candidate": event["candidate"],
        "signal_time": event["signal_time"],
        "entry_time": event["entry_time"],
        "binance_symbol": "",
        "mapping_status": "unlisted",
    }
    for window in EVENT_WINDOWS:
        row[f"{window}_covered"] = False
        row[f"{window}_coverage_ratio"] = 0.0
        row[f"{window}_source_quality"] = "unlisted"
    return row


def _event_row(
    event: pd.Series,
    binance_symbol: str,
    day_frames: dict[date, pd.DataFrame],
    cfg: OrderflowHistoryConfig,
) -> dict[str, object]:
    span_days = _event_day_span(event)
    frames = [day_frames[day] for day in span_days if day in day_frames]
    span = (
        pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if frames
        else _empty_normalized()
    )
    row: dict[str, object] = {
        "signal_id": event["signal_id"],
        "exchange": event["exchange"],
        "symbol": event["symbol"],
        "candidate": event["candidate"],
        "signal_time": event["signal_time"],
        "entry_time": event["entry_time"],
        "binance_symbol": binance_symbol,
        "mapping_status": "mapped" if frames else "mapped_no_tape",
    }
    windows = event_windows(event["signal_time"], event["entry_time"])
    for window_name, (start, end) in windows.items():
        if end <= start:
            stats = {
                "covered": False,
                "coverage_ratio": 0.0,
                "source_quality": "empty_window",
            }
            row.update({f"{window_name}_{key}": value for key, value in stats.items()})
            continue
        stats = summarize_trade_window(
            span,
            start,
            end,
            cfg.large_trade_quantile,
            assume_normalized=True,
        )
        stats.pop("window_start", None)
        stats.pop("window_end", None)
        stats.pop("cache_start", None)
        stats.pop("cache_end", None)
        row.update({f"{window_name}_{key}": value for key, value in stats.items()})
    return row


def write_cic_event_orderflow(cfg: OrderflowHistoryConfig | None = None) -> dict[str, Path]:
    """End-to-end backfill: trade cache -> events -> archives -> window features."""
    cfg = cfg or OrderflowHistoryConfig()
    if not cfg.trade_cache_path.exists():
        raise FileNotFoundError(
            f"capacity trade cache not found: {cfg.trade_cache_path} (run run-v09d first)"
        )
    trades = read_parquet(cfg.trade_cache_path)
    events = extract_events(trades, cfg.candidates, cfg.max_events)
    features = build_event_orderflow(events, cfg)
    report_root = ensure_dir(cfg.report_root)
    output_path = cfg.history_root / "cic_event_orderflow.parquet"
    coverage_path = report_root / "orderflow_history_coverage.csv"
    if not features.empty:
        write_parquet(features, output_path)
        coverage = _coverage_summary(features)
        coverage.to_csv(coverage_path, index=False)
    return {"event_orderflow": output_path, "coverage": coverage_path}


def _coverage_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, sample in features.groupby("candidate", dropna=False):
        row: dict[str, object] = {
            "candidate": candidate,
            "events": int(len(sample)),
            "mapped": int(sample["mapping_status"].astype(str).eq("mapped").sum()),
            "unlisted": int(sample["mapping_status"].astype(str).eq("unlisted").sum()),
        }
        for window in EVENT_WINDOWS:
            covered = sample.get(f"{window}_covered", pd.Series(dtype=bool)).fillna(False).astype(bool)
            row[f"{window}_covered_rate"] = float(covered.mean()) if len(covered) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_CANDIDATES",
    "EVENT_WINDOWS",
    "HISTORY_ROOT",
    "PRE_ENTRY_WINDOWS",
    "REPORT_ROOT",
    "OrderflowHistoryConfig",
    "aggtrades_daily_url",
    "binance_symbol_candidates",
    "build_event_orderflow",
    "event_windows",
    "extract_events",
    "load_aggtrades_zip",
    "normalize_aggtrades",
    "signal_event_id",
    "write_cic_event_orderflow",
]
