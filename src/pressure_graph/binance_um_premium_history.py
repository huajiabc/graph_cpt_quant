"""Historical Binance USD-M premium-index klines from public archives."""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
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
from pressure_graph.orderflow_history import binance_symbol_candidates


BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data"
INTERVAL = "15m"


@dataclass(frozen=True)
class BinancePremiumConfig:
    output_root: Path = Path("data/external/binance_um_premium_15m")
    download_workers: int = 12
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_sleep_seconds: float = 1.0
    mapping_probe_day: date = date(2026, 6, 1)
    verify_checksums: bool = True
    merge_existing: bool = True


@dataclass(frozen=True)
class BinancePremiumSymbolResult:
    bybit_symbol: str
    binance_symbol: str | None
    requested_archives: int
    downloaded_archives: int
    verified_archives: int
    missing_archives: int
    rows: int
    first_time: str | None
    last_time: str | None
    output_path: str | None
    error: str | None = None


def monthly_premium_url(binance_symbol: str, month: date) -> str:
    stamp = month.strftime("%Y-%m")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/monthly/premiumIndexKlines/"
        f"{binance_symbol}/{INTERVAL}/{binance_symbol}-{INTERVAL}-{stamp}.zip"
    )


def daily_premium_url(binance_symbol: str, day: date) -> str:
    stamp = day.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/daily/premiumIndexKlines/"
        f"{binance_symbol}/{INTERVAL}/{binance_symbol}-{INTERVAL}-{stamp}.zip"
    )


def _request_bytes(
    url: str,
    client: httpx.Client,
    cfg: BinancePremiumConfig,
    *,
    method: str = "GET",
) -> bytes | None:
    for attempt in range(cfg.retry_attempts):
        try:
            response = client.request(method, url)
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


def verify_archive_checksum(content: bytes, checksum_content: bytes) -> bool:
    text = checksum_content.decode("utf-8").strip()
    expected = text.split()[0].lower() if text else ""
    actual = hashlib.sha256(content).hexdigest()
    return bool(expected) and actual == expected


def resolve_binance_premium_symbol(
    bybit_symbol: str,
    cfg: BinancePremiumConfig = BinancePremiumConfig(),
    client: httpx.Client | None = None,
) -> str | None:
    owns_client = client is None
    session = client or httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    )
    try:
        for candidate in binance_symbol_candidates(bybit_symbol):
            if (
                _request_bytes(
                    daily_premium_url(candidate, cfg.mapping_probe_day),
                    session,
                    cfg,
                    method="HEAD",
                )
                is not None
            ):
                return candidate
        return None
    finally:
        if owns_client:
            session.close()


def parse_premium_zip(
    content: bytes,
    bybit_symbol: str,
    binance_symbol: str,
    source_archive: str,
) -> pd.DataFrame:
    """Parse one headerless premium-index kline archive."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(
                f"Expected one CSV in premium archive, found {csv_names!r}"
            )
        frame = pd.read_csv(
            archive.open(csv_names[0]), header=None, names=KLINE_COLUMNS
        )
    frame["bar_open_time"] = _timestamp(frame["open_time_raw"])
    frame["source_close_time"] = _timestamp(frame["close_time_raw"])
    frame["feature_time"] = frame["bar_open_time"] + pd.Timedelta(minutes=15)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bybit_symbol"] = bybit_symbol.upper().strip()
    frame["binance_symbol"] = binance_symbol
    frame["source_archive"] = source_archive
    wanted = [
        "bybit_symbol",
        "binance_symbol",
        "source_archive",
        "bar_open_time",
        "source_close_time",
        "feature_time",
        "open",
        "high",
        "low",
        "close",
    ]
    return (
        frame.loc[:, wanted]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["bar_open_time", "feature_time", "close"])
        .sort_values("bar_open_time")
        .drop_duplicates(["bybit_symbol", "bar_open_time"], keep="last")
        .reset_index(drop=True)
    )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _merge_existing(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return frame
    existing = pd.read_parquet(path)
    return (
        pd.concat([existing, frame], ignore_index=True)
        .drop_duplicates(["bybit_symbol", "feature_time"], keep="last")
        .sort_values("feature_time")
        .reset_index(drop=True)
    )


def download_premium_symbol(
    bybit_symbol: str,
    start_day: date,
    end_day: date,
    cfg: BinancePremiumConfig = BinancePremiumConfig(),
) -> BinancePremiumSymbolResult:
    output_path = cfg.output_root / f"{bybit_symbol.upper().strip()}.parquet"
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        binance_symbol = resolve_binance_premium_symbol(bybit_symbol, cfg, client)
        if binance_symbol is None:
            return BinancePremiumSymbolResult(
                bybit_symbol=bybit_symbol,
                binance_symbol=None,
                requested_archives=0,
                downloaded_archives=0,
                verified_archives=0,
                missing_archives=0,
                rows=0,
                first_time=None,
                last_time=None,
                output_path=None,
                error="unresolved_binance_symbol",
            )
        months = _month_starts(start_day, end_day)
        days = _partial_days(start_day, end_day)
        urls = [monthly_premium_url(binance_symbol, month) for month in months]
        urls.extend(daily_premium_url(binance_symbol, day) for day in days)
        frames: list[pd.DataFrame] = []
        missing = 0
        verified = 0

        def download_url(url: str) -> None:
            nonlocal missing, verified
            content = _request_bytes(url, client, cfg)
            if content is None:
                missing += 1
                return
            if cfg.verify_checksums:
                checksum = _request_bytes(f"{url}.CHECKSUM", client, cfg)
                if checksum is None:
                    raise ValueError(f"Missing checksum for {url}")
                if not verify_archive_checksum(content, checksum):
                    raise ValueError(f"Checksum mismatch for {url}")
                verified += 1
            frames.append(
                parse_premium_zip(
                    content,
                    bybit_symbol,
                    binance_symbol,
                    url.rsplit("/", maxsplit=1)[-1],
                )
            )

        for url in urls:
            download_url(url)
        if frames:
            provisional = pd.concat(frames, ignore_index=True)
            observed = pd.DatetimeIndex(provisional["bar_open_time"].dropna().unique())
            expected = pd.date_range(
                observed.min().floor("15min"),
                observed.max().floor("15min"),
                freq="15min",
                tz="UTC",
            )
            repair_days = sorted({timestamp.date() for timestamp in expected.difference(observed)})
            for repair_day in repair_days:
                repair_url = daily_premium_url(binance_symbol, repair_day)
                if repair_url in urls:
                    continue
                urls.append(repair_url)
                download_url(repair_url)
    if not frames:
        return BinancePremiumSymbolResult(
            bybit_symbol=bybit_symbol,
            binance_symbol=binance_symbol,
            requested_archives=len(urls),
            downloaded_archives=0,
            verified_archives=verified,
            missing_archives=missing,
            rows=0,
            first_time=None,
            last_time=None,
            output_path=None,
            error="no_premium_archives",
        )
    frame = (
        pd.concat(frames, ignore_index=True)
        .loc[
            lambda x: x["bar_open_time"].ge(pd.Timestamp(start_day, tz="UTC"))
            & x["bar_open_time"].lt(
                pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1)
            )
        ]
        .drop_duplicates(["bybit_symbol", "feature_time"], keep="last")
        .sort_values("feature_time")
        .reset_index(drop=True)
    )
    if cfg.merge_existing:
        frame = _merge_existing(frame, output_path)
    _atomic_write_parquet(frame, output_path)
    return BinancePremiumSymbolResult(
        bybit_symbol=bybit_symbol,
        binance_symbol=binance_symbol,
        requested_archives=len(urls),
        downloaded_archives=len(frames),
        verified_archives=verified,
        missing_archives=missing,
        rows=len(frame),
        first_time=frame["feature_time"].min().isoformat(),
        last_time=frame["feature_time"].max().isoformat(),
        output_path=str(output_path),
        error=None,
    )


def inventory_binance_premium(
    output_root: Path,
    expected_symbols: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    covered: set[str] = set()
    for path in sorted(output_root.glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "bybit_symbol",
                "binance_symbol",
                "source_archive",
                "feature_time",
            ],
        )
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], utc=True, errors="coerce"
        )
        symbol = path.stem.upper()
        valid_time = frame["feature_time"].dropna()
        archives = frame["source_archive"].dropna().astype(str).nunique()
        bybit = frame["bybit_symbol"].dropna().astype(str)
        binance = frame["binance_symbol"].dropna().astype(str)
        rows.append(
            {
                "bybit_symbol": bybit.iloc[0] if len(bybit) else symbol,
                "binance_symbol": binance.iloc[0] if len(binance) else None,
                "source_archives": int(archives),
                "rows": len(frame),
                "first_time": (
                    valid_time.min().isoformat() if not valid_time.empty else None
                ),
                "last_time": (
                    valid_time.max().isoformat() if not valid_time.empty else None
                ),
                "output_path": str(path),
                "error": None,
            }
        )
        covered.add(symbol)
    for symbol in sorted(set(expected_symbols or []) - covered):
        rows.append(
            {
                "bybit_symbol": symbol,
                "binance_symbol": None,
                "source_archives": 0,
                "rows": 0,
                "first_time": None,
                "last_time": None,
                "output_path": None,
                "error": "missing_data_file",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("bybit_symbol").reset_index(drop=True)


def write_premium_inventory(
    cfg: BinancePremiumConfig,
    expected_symbols: list[str] | None = None,
    start_day: date | None = None,
    end_day: date | None = None,
) -> pd.DataFrame:
    inventory = inventory_binance_premium(cfg.output_root, expected_symbols)
    root = ensure_dir(cfg.output_root)
    inventory.to_csv(root / "manifest.csv", index=False)
    coverage = {
        "start_day": start_day.isoformat() if start_day else None,
        "end_day": end_day.isoformat() if end_day else None,
        "requested_symbols": len(expected_symbols or inventory),
        "covered_symbols": int(inventory["rows"].gt(0).sum()) if len(inventory) else 0,
        "rows": int(inventory["rows"].sum()) if len(inventory) else 0,
        "config": {**asdict(cfg), "output_root": str(cfg.output_root)},
    }
    (root / "coverage.json").write_text(
        json.dumps(coverage, indent=2, default=str), encoding="utf-8"
    )
    return inventory


def backfill_binance_premium(
    symbols: list[str],
    start_day: date,
    end_day: date,
    cfg: BinancePremiumConfig = BinancePremiumConfig(),
) -> pd.DataFrame:
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    prior_expected: list[str] = []
    existing_manifest = cfg.output_root / "manifest.csv"
    if existing_manifest.exists():
        prior = pd.read_csv(existing_manifest, usecols=["bybit_symbol"])
        prior_expected = prior["bybit_symbol"].dropna().astype(str).tolist()
    expected = sorted(set(prior_expected) | set(normalized))
    results: list[BinancePremiumSymbolResult] = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {
            executor.submit(
                download_premium_symbol, symbol, start_day, end_day, cfg
            ): symbol
            for symbol in normalized
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    BinancePremiumSymbolResult(
                        bybit_symbol=symbol,
                        binance_symbol=None,
                        requested_archives=0,
                        downloaded_archives=0,
                        verified_archives=0,
                        missing_archives=0,
                        rows=0,
                        first_time=None,
                        last_time=None,
                        output_path=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    call_manifest = pd.DataFrame(asdict(result) for result in results).sort_values(
        "bybit_symbol"
    )
    ensure_dir(cfg.output_root)
    call_manifest.to_csv(cfg.output_root / "last_download.csv", index=False)
    write_premium_inventory(cfg, expected, start_day, end_day)
    return call_manifest.reset_index(drop=True)


__all__ = [
    "BinancePremiumConfig",
    "BinancePremiumSymbolResult",
    "backfill_binance_premium",
    "daily_premium_url",
    "download_premium_symbol",
    "inventory_binance_premium",
    "monthly_premium_url",
    "parse_premium_zip",
    "resolve_binance_premium_symbol",
    "verify_archive_checksum",
    "write_premium_inventory",
]
