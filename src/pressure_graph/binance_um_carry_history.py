"""Binance USD-M klines and funding history for same-coin carry research."""
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

from pressure_graph.binance_spot_history import (
    KLINE_COLUMNS,
    NUMERIC_COLUMNS,
    _month_starts,
    _partial_days,
    _timestamp,
)
from pressure_graph.io import ensure_dir


BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data"
BINANCE_FAPI_BASE = "https://fapi.binance.com"
UM_ALIASES = {"SHIB1000USDT": "1000SHIBUSDT"}


@dataclass(frozen=True)
class BinanceUmCarryConfig:
    output_root: Path = Path("data/external/binance_um_carry")
    download_workers: int = 12
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_sleep_seconds: float = 1.0


@dataclass(frozen=True)
class BinanceUmCarrySymbolResult:
    bybit_symbol: str
    binance_symbol: str
    requested_kline_archives: int
    downloaded_kline_archives: int
    missing_kline_archives: int
    kline_rows: int
    funding_rows: int
    first_kline_time: str | None
    last_kline_time: str | None
    first_funding_time: str | None
    last_funding_time: str | None
    error: str | None = None


def um_symbol(bybit_symbol: str) -> str:
    normalized = bybit_symbol.upper().strip()
    return UM_ALIASES.get(normalized, normalized)


def monthly_um_kline_url(binance_symbol: str, month: date) -> str:
    stamp = month.strftime("%Y-%m")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/monthly/klines/"
        f"{binance_symbol}/1h/{binance_symbol}-1h-{stamp}.zip"
    )


def daily_um_kline_url(binance_symbol: str, day: date) -> str:
    stamp = day.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/daily/klines/"
        f"{binance_symbol}/1h/{binance_symbol}-1h-{stamp}.zip"
    )


def parse_um_kline_zip(
    content: bytes, bybit_symbol: str, binance_symbol: str | None = None
) -> pd.DataFrame:
    resolved = binance_symbol or um_symbol(bybit_symbol)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in USD-M archive, found {csv_names!r}")
        frame = pd.read_csv(archive.open(csv_names[0]), header=None, names=KLINE_COLUMNS)
    frame["bar_open_time"] = _timestamp(frame["open_time_raw"])
    frame["source_close_time"] = _timestamp(frame["close_time_raw"])
    frame["feature_time"] = frame["bar_open_time"] + pd.Timedelta(hours=1)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bybit_symbol"] = bybit_symbol.upper().strip()
    frame["binance_symbol"] = resolved
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


def parse_funding_payload(
    payload: list[dict[str, object]], bybit_symbol: str, binance_symbol: str
) -> pd.DataFrame:
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame()
    required = {"fundingTime", "fundingRate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Funding payload is missing columns: {sorted(missing)}")
    frame["funding_time"] = pd.to_datetime(
        pd.to_numeric(frame["fundingTime"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    ).dt.floor("s")
    frame["funding_rate_settled"] = pd.to_numeric(
        frame["fundingRate"], errors="coerce"
    )
    frame["bybit_symbol"] = bybit_symbol.upper().strip()
    frame["binance_symbol"] = binance_symbol
    return (
        frame[
            [
                "bybit_symbol",
                "binance_symbol",
                "funding_time",
                "funding_rate_settled",
            ]
        ]
        .dropna(subset=["funding_time", "funding_rate_settled"])
        .drop_duplicates(["bybit_symbol", "funding_time"], keep="last")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


def _get_bytes(url: str, client: httpx.Client, cfg: BinanceUmCarryConfig) -> bytes | None:
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


def fetch_um_funding(
    bybit_symbol: str,
    start_day: date,
    end_day: date,
    client: httpx.Client,
    cfg: BinanceUmCarryConfig,
) -> pd.DataFrame:
    binance_symbol = um_symbol(bybit_symbol)
    cursor = int(pd.Timestamp(start_day, tz="UTC").timestamp() * 1000)
    end_ms = int(
        pd.Timestamp(end_day + timedelta(days=1), tz="UTC").timestamp() * 1000 - 1
    )
    payload: list[dict[str, object]] = []
    while cursor <= end_ms:
        response = client.get(
            f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
            params={
                "symbol": binance_symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if response.status_code == 400 and response.json().get("code") == -1121:
            return pd.DataFrame()
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        payload.extend(batch)
        latest = max(int(row["fundingTime"]) for row in batch)
        if latest < cursor or len(batch) < 1000:
            break
        cursor = latest + 1
    return parse_funding_payload(payload, bybit_symbol, binance_symbol)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def download_um_carry_symbol(
    bybit_symbol: str,
    kline_start_day: date,
    funding_start_day: date,
    end_day: date,
    cfg: BinanceUmCarryConfig = BinanceUmCarryConfig(),
) -> BinanceUmCarrySymbolResult:
    binance_symbol = um_symbol(bybit_symbol)
    months = _month_starts(kline_start_day, end_day)
    days = _partial_days(kline_start_day, end_day)
    urls = [monthly_um_kline_url(binance_symbol, month) for month in months]
    urls.extend(daily_um_kline_url(binance_symbol, day) for day in days)
    kline_frames: list[pd.DataFrame] = []
    missing = 0
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        for url in urls:
            content = _get_bytes(url, client, cfg)
            if content is None:
                missing += 1
                continue
            kline_frames.append(
                parse_um_kline_zip(content, bybit_symbol, binance_symbol)
            )
        funding = fetch_um_funding(
            bybit_symbol, funding_start_day, end_day, client, cfg
        )
    kline = pd.DataFrame()
    if kline_frames:
        kline = (
            pd.concat(kline_frames, ignore_index=True)
            .loc[
                lambda x: x["feature_time"].ge(
                    pd.Timestamp(kline_start_day, tz="UTC")
                )
            ]
            .loc[
                lambda x: x["feature_time"].lt(
                    pd.Timestamp(end_day + timedelta(days=1), tz="UTC")
                )
            ]
            .drop_duplicates(["bybit_symbol", "feature_time"], keep="last")
            .sort_values("feature_time")
            .reset_index(drop=True)
        )
        _atomic_write_parquet(
            kline, cfg.output_root / "klines_1h" / f"{bybit_symbol}.parquet"
        )
    if not funding.empty:
        _atomic_write_parquet(
            funding, cfg.output_root / "funding" / f"{bybit_symbol}.parquet"
        )
    errors = []
    if kline.empty:
        errors.append("no_um_klines")
    if funding.empty:
        errors.append("no_um_funding")
    return BinanceUmCarrySymbolResult(
        bybit_symbol=bybit_symbol,
        binance_symbol=binance_symbol,
        requested_kline_archives=len(urls),
        downloaded_kline_archives=len(kline_frames),
        missing_kline_archives=missing,
        kline_rows=len(kline),
        funding_rows=len(funding),
        first_kline_time=(kline["feature_time"].min().isoformat() if not kline.empty else None),
        last_kline_time=(kline["feature_time"].max().isoformat() if not kline.empty else None),
        first_funding_time=(funding["funding_time"].min().isoformat() if not funding.empty else None),
        last_funding_time=(funding["funding_time"].max().isoformat() if not funding.empty else None),
        error="|".join(errors) or None,
    )


def backfill_binance_um_carry(
    symbols: list[str],
    kline_start_day: date,
    funding_start_day: date,
    end_day: date,
    cfg: BinanceUmCarryConfig = BinanceUmCarryConfig(),
) -> pd.DataFrame:
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    results: list[BinanceUmCarrySymbolResult] = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {
            executor.submit(
                download_um_carry_symbol,
                symbol,
                kline_start_day,
                funding_start_day,
                end_day,
                cfg,
            ): symbol
            for symbol in normalized
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = BinanceUmCarrySymbolResult(
                    bybit_symbol=symbol,
                    binance_symbol=um_symbol(symbol),
                    requested_kline_archives=0,
                    downloaded_kline_archives=0,
                    missing_kline_archives=0,
                    kline_rows=0,
                    funding_rows=0,
                    first_kline_time=None,
                    last_kline_time=None,
                    first_funding_time=None,
                    last_funding_time=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            print(
                f"um-carry: {completed}/{len(normalized)} {symbol} "
                f"klines={result.kline_rows} funding={result.funding_rows} "
                f"error={result.error or '-'}",
                flush=True,
            )
    manifest = pd.DataFrame([asdict(result) for result in results]).sort_values(
        "bybit_symbol"
    )
    ensure_dir(cfg.output_root)
    manifest.to_csv(cfg.output_root / "manifest.csv", index=False)
    coverage = {
        "kline_start_day": kline_start_day.isoformat(),
        "funding_start_day": funding_start_day.isoformat(),
        "end_day": end_day.isoformat(),
        "requested_symbols": len(normalized),
        "covered_symbols": int(
            (manifest["kline_rows"].gt(0) & manifest["funding_rows"].gt(0)).sum()
        ),
        "kline_rows": int(manifest["kline_rows"].sum()),
        "funding_rows": int(manifest["funding_rows"].sum()),
        "config": {**asdict(cfg), "output_root": str(cfg.output_root)},
    }
    (cfg.output_root / "coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    return manifest.reset_index(drop=True)
