"""Checksummed Binance USD-M mark/index price kline history."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd

from pressure_graph.binance_spot_history import _month_starts, _partial_days
from pressure_graph.binance_um_premium_history import (
    BINANCE_PUBLIC_DATA_BASE,
    INTERVAL,
    _atomic_write_parquet,
    _request_bytes,
    parse_premium_zip,
    verify_archive_checksum,
)
from pressure_graph.io import ensure_dir
from pressure_graph.orderflow_history import binance_symbol_candidates


ReferenceDataset = Literal["markPriceKlines", "indexPriceKlines"]


@dataclass(frozen=True)
class BinanceReferenceConfig:
    dataset: ReferenceDataset
    output_root: Path
    download_workers: int = 12
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_sleep_seconds: float = 1.0
    mapping_probe_day: date = date(2026, 6, 1)
    verify_checksums: bool = True
    merge_existing: bool = True


@dataclass(frozen=True)
class BinanceReferenceSymbolResult:
    dataset: str
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


def monthly_reference_url(
    dataset: ReferenceDataset, binance_symbol: str, month: date
) -> str:
    stamp = month.strftime("%Y-%m")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/monthly/{dataset}/"
        f"{binance_symbol}/{INTERVAL}/{binance_symbol}-{INTERVAL}-{stamp}.zip"
    )


def daily_reference_url(
    dataset: ReferenceDataset, binance_symbol: str, day: date
) -> str:
    stamp = day.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/daily/{dataset}/"
        f"{binance_symbol}/{INTERVAL}/{binance_symbol}-{INTERVAL}-{stamp}.zip"
    )


def resolve_binance_reference_symbol(
    bybit_symbol: str,
    cfg: BinanceReferenceConfig,
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
                    daily_reference_url(cfg.dataset, candidate, cfg.mapping_probe_day),
                    session,
                    cfg,  # type: ignore[arg-type]
                    method="HEAD",
                )
                is not None
            ):
                return candidate
        return None
    finally:
        if owns_client:
            session.close()


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


def download_reference_symbol(
    bybit_symbol: str,
    start_day: date,
    end_day: date,
    cfg: BinanceReferenceConfig,
) -> BinanceReferenceSymbolResult:
    output_path = cfg.output_root / f"{bybit_symbol.upper().strip()}.parquet"
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        binance_symbol = resolve_binance_reference_symbol(bybit_symbol, cfg, client)
        if binance_symbol is None:
            return BinanceReferenceSymbolResult(
                dataset=cfg.dataset,
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
        urls = [
            monthly_reference_url(cfg.dataset, binance_symbol, month)
            for month in months
        ]
        urls.extend(
            daily_reference_url(cfg.dataset, binance_symbol, day) for day in days
        )
        frames: list[pd.DataFrame] = []
        missing = 0
        verified = 0

        def download_url(url: str) -> None:
            nonlocal missing, verified
            content = _request_bytes(client=client, url=url, cfg=cfg)  # type: ignore[arg-type]
            if content is None:
                missing += 1
                return
            if cfg.verify_checksums:
                checksum = _request_bytes(
                    client=client, url=f"{url}.CHECKSUM", cfg=cfg  # type: ignore[arg-type]
                )
                if checksum is None:
                    raise ValueError(f"Missing checksum for {url}")
                if not verify_archive_checksum(content, checksum):
                    raise ValueError(f"Checksum mismatch for {url}")
                verified += 1
            frame = parse_premium_zip(
                content,
                bybit_symbol,
                binance_symbol,
                url.rsplit("/", maxsplit=1)[-1],
            )
            frame["dataset"] = cfg.dataset
            frames.append(frame)

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
            repair_days = sorted(
                {timestamp.date() for timestamp in expected.difference(observed)}
            )
            for repair_day in repair_days:
                repair_url = daily_reference_url(
                    cfg.dataset, binance_symbol, repair_day
                )
                if repair_url in urls:
                    continue
                urls.append(repair_url)
                download_url(repair_url)
    if not frames:
        return BinanceReferenceSymbolResult(
            dataset=cfg.dataset,
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
            error="no_reference_archives",
        )
    frame = (
        pd.concat(frames, ignore_index=True)
        .loc[
            lambda value: value["bar_open_time"].ge(
                pd.Timestamp(start_day, tz="UTC")
            )
            & value["bar_open_time"].lt(
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
    return BinanceReferenceSymbolResult(
        dataset=cfg.dataset,
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


def inventory_binance_reference(
    cfg: BinanceReferenceConfig,
    expected_symbols: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    covered: set[str] = set()
    for path in sorted(cfg.output_root.glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "bybit_symbol",
                "binance_symbol",
                "source_archive",
                "feature_time",
            ],
        )
        frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True)
        symbol = path.stem.upper()
        rows.append(
            {
                "dataset": cfg.dataset,
                "bybit_symbol": str(frame["bybit_symbol"].dropna().iloc[0]),
                "binance_symbol": str(frame["binance_symbol"].dropna().iloc[0]),
                "source_archives": int(frame["source_archive"].nunique()),
                "rows": len(frame),
                "first_time": frame["feature_time"].min().isoformat(),
                "last_time": frame["feature_time"].max().isoformat(),
                "output_path": str(path),
                "error": None,
            }
        )
        covered.add(symbol)
    for symbol in sorted(set(expected_symbols or []) - covered):
        rows.append(
            {
                "dataset": cfg.dataset,
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
    return pd.DataFrame(rows).sort_values("bybit_symbol").reset_index(drop=True)


def write_reference_inventory(
    cfg: BinanceReferenceConfig,
    expected_symbols: list[str] | None = None,
    start_day: date | None = None,
    end_day: date | None = None,
) -> pd.DataFrame:
    inventory = inventory_binance_reference(cfg, expected_symbols)
    root = ensure_dir(cfg.output_root)
    inventory.to_csv(root / "manifest.csv", index=False)
    coverage = {
        "dataset": cfg.dataset,
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


def backfill_binance_reference(
    symbols: list[str],
    start_day: date,
    end_day: date,
    cfg: BinanceReferenceConfig,
) -> pd.DataFrame:
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    prior_expected: list[str] = []
    existing_manifest = cfg.output_root / "manifest.csv"
    if existing_manifest.exists():
        prior = pd.read_csv(existing_manifest, usecols=["bybit_symbol"])
        prior_expected = prior["bybit_symbol"].dropna().astype(str).tolist()
    expected = sorted(set(prior_expected) | set(normalized))
    results: list[BinanceReferenceSymbolResult] = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {
            executor.submit(
                download_reference_symbol, symbol, start_day, end_day, cfg
            ): symbol
            for symbol in normalized
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    BinanceReferenceSymbolResult(
                        dataset=cfg.dataset,
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
    write_reference_inventory(cfg, expected, start_day, end_day)
    return call_manifest.reset_index(drop=True)


__all__ = [
    "BinanceReferenceConfig",
    "BinanceReferenceSymbolResult",
    "ReferenceDataset",
    "backfill_binance_reference",
    "daily_reference_url",
    "download_reference_symbol",
    "inventory_binance_reference",
    "monthly_reference_url",
    "resolve_binance_reference_symbol",
    "write_reference_inventory",
]
