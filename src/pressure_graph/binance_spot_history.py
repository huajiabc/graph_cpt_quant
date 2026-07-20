"""Binance public spot one-hour kline history for spot/perpetual carry research."""
from __future__ import annotations

import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data"
SPOT_ALIASES = {
    "1000BONKUSDT": "BONKUSDT",
    "1000PEPEUSDT": "PEPEUSDT",
    "SHIB1000USDT": "SHIBUSDT",
}
KLINE_COLUMNS = (
    "open_time_raw",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_raw",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)
NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)


@dataclass(frozen=True)
class BinanceSpotConfig:
    output_root: Path = Path("data/external/binance_spot_1h")
    download_workers: int = 12
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_sleep_seconds: float = 1.0


@dataclass(frozen=True)
class BinanceSpotSymbolResult:
    bybit_symbol: str
    binance_symbol: str
    requested_archives: int
    downloaded_archives: int
    missing_archives: int
    rows: int
    first_time: str | None
    last_time: str | None
    output_path: str | None
    error: str | None = None


def spot_symbol(bybit_symbol: str) -> str:
    normalized = bybit_symbol.upper().strip()
    return SPOT_ALIASES.get(normalized, normalized)


def monthly_kline_url(binance_symbol: str, month: date) -> str:
    stamp = month.strftime("%Y-%m")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/spot/monthly/klines/{binance_symbol}/1h/"
        f"{binance_symbol}-1h-{stamp}.zip"
    )


def daily_kline_url(binance_symbol: str, day: date) -> str:
    stamp = day.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/spot/daily/klines/{binance_symbol}/1h/"
        f"{binance_symbol}-1h-{stamp}.zip"
    )


def _timestamp(values: pd.Series) -> pd.Series:
    """Decode Binance's historical millisecond and newer microsecond epochs."""
    numeric = pd.to_numeric(values, errors="coerce")
    milliseconds = numeric.abs().lt(100_000_000_000_000)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if milliseconds.any():
        result.loc[milliseconds] = pd.to_datetime(
            numeric.loc[milliseconds], unit="ms", utc=True, errors="coerce"
        )
    if (~milliseconds).any():
        result.loc[~milliseconds] = pd.to_datetime(
            numeric.loc[~milliseconds], unit="us", utc=True, errors="coerce"
        )
    return result


def parse_spot_kline_zip(content: bytes, bybit_symbol: str) -> pd.DataFrame:
    """Parse one headerless public archive and expose strictly closed-bar times."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in spot archive, found {csv_names!r}")
        frame = pd.read_csv(archive.open(csv_names[0]), header=None, names=KLINE_COLUMNS)
    frame["bar_open_time"] = _timestamp(frame["open_time_raw"])
    frame["source_close_time"] = _timestamp(frame["close_time_raw"])
    frame["feature_time"] = frame["bar_open_time"] + pd.Timedelta(hours=1)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bybit_symbol"] = bybit_symbol.upper().strip()
    frame["binance_symbol"] = spot_symbol(bybit_symbol)
    wanted = [
        "bybit_symbol",
        "binance_symbol",
        "bar_open_time",
        "source_close_time",
        "feature_time",
        *NUMERIC_COLUMNS,
    ]
    return (
        frame.loc[:, wanted]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["bar_open_time", "feature_time", "close"])
        .sort_values("bar_open_time")
        .drop_duplicates(["bybit_symbol", "bar_open_time"], keep="last")
        .reset_index(drop=True)
    )


def _month_starts(start_day: date, end_day: date) -> list[date]:
    cursor = date(start_day.year, start_day.month, 1)
    months = []
    while True:
        next_month = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
        if next_month - timedelta(days=1) > end_day:
            break
        months.append(cursor)
        cursor = next_month
    return months


def _partial_days(start_day: date, end_day: date) -> list[date]:
    complete_months = _month_starts(start_day, end_day)
    if complete_months:
        last = complete_months[-1]
        cursor = date(last.year + (last.month == 12), 1 if last.month == 12 else last.month + 1, 1)
    else:
        cursor = start_day
    cursor = max(cursor, start_day)
    return [
        cursor + timedelta(days=offset)
        for offset in range((end_day - cursor).days + 1)
    ] if cursor <= end_day else []


def _request_bytes(url: str, client: httpx.Client, cfg: BinanceSpotConfig) -> bytes | None:
    for attempt in range(cfg.retry_attempts):
        try:
            response = client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500 or attempt + 1 >= cfg.retry_attempts:
                raise
        except httpx.TransportError:
            if attempt + 1 >= cfg.retry_attempts:
                raise
        time.sleep(cfg.retry_sleep_seconds * (attempt + 1))
    return None


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def download_spot_symbol(
    bybit_symbol: str,
    start_day: date,
    end_day: date,
    cfg: BinanceSpotConfig = BinanceSpotConfig(),
) -> BinanceSpotSymbolResult:
    binance_symbol = spot_symbol(bybit_symbol)
    months = _month_starts(start_day, end_day)
    days = _partial_days(start_day, end_day)
    urls = [monthly_kline_url(binance_symbol, month) for month in months]
    urls.extend(daily_kline_url(binance_symbol, day) for day in days)
    frames: list[pd.DataFrame] = []
    missing = 0
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        for url in urls:
            content = _request_bytes(url, client, cfg)
            if content is None:
                missing += 1
                continue
            frames.append(parse_spot_kline_zip(content, bybit_symbol))
    if not frames:
        return BinanceSpotSymbolResult(
            bybit_symbol=bybit_symbol,
            binance_symbol=binance_symbol,
            requested_archives=len(urls),
            downloaded_archives=0,
            missing_archives=missing,
            rows=0,
            first_time=None,
            last_time=None,
            output_path=None,
            error="no_spot_archives",
        )
    frame = (
        pd.concat(frames, ignore_index=True)
        .loc[lambda x: x["feature_time"].ge(pd.Timestamp(start_day, tz="UTC"))]
        .loc[
            lambda x: x["feature_time"].lt(
                pd.Timestamp(end_day + timedelta(days=1), tz="UTC")
            )
        ]
        .drop_duplicates(["bybit_symbol", "feature_time"], keep="last")
        .sort_values("feature_time")
        .reset_index(drop=True)
    )
    output_path = cfg.output_root / f"{bybit_symbol}.parquet"
    _atomic_write_parquet(frame, output_path)
    return BinanceSpotSymbolResult(
        bybit_symbol=bybit_symbol,
        binance_symbol=binance_symbol,
        requested_archives=len(urls),
        downloaded_archives=len(frames),
        missing_archives=missing,
        rows=len(frame),
        first_time=frame["feature_time"].min().isoformat(),
        last_time=frame["feature_time"].max().isoformat(),
        output_path=str(output_path),
    )


def backfill_binance_spot(
    symbols: list[str],
    start_day: date,
    end_day: date,
    cfg: BinanceSpotConfig = BinanceSpotConfig(),
) -> pd.DataFrame:
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    results: list[BinanceSpotSymbolResult] = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {
            executor.submit(download_spot_symbol, symbol, start_day, end_day, cfg): symbol
            for symbol in normalized
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = BinanceSpotSymbolResult(
                    bybit_symbol=symbol,
                    binance_symbol=spot_symbol(symbol),
                    requested_archives=0,
                    downloaded_archives=0,
                    missing_archives=0,
                    rows=0,
                    first_time=None,
                    last_time=None,
                    output_path=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            print(
                f"spot: {completed}/{len(normalized)} {symbol} "
                f"archives={result.downloaded_archives} rows={result.rows} "
                f"error={result.error or '-'}",
                flush=True,
            )
    manifest = pd.DataFrame([asdict(result) for result in results]).sort_values(
        "bybit_symbol"
    )
    ensure_dir(cfg.output_root)
    manifest.to_csv(cfg.output_root / "manifest.csv", index=False)
    coverage = {
        "start_day": start_day.isoformat(),
        "end_day": end_day.isoformat(),
        "requested_symbols": len(normalized),
        "covered_symbols": int(manifest["rows"].gt(0).sum()),
        "rows": int(manifest["rows"].sum()),
        "config": {**asdict(cfg), "output_root": str(cfg.output_root)},
    }
    (cfg.output_root / "coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    return manifest.reset_index(drop=True)
