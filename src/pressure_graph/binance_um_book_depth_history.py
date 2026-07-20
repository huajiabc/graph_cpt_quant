"""Daily features from official Binance USD-M public book-depth archives."""

from __future__ import annotations

import hashlib
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
from pressure_graph.orderflow_history import binance_symbol_candidates


BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data"
BOOK_DEPTH_COLUMNS = ("timestamp", "percentage", "depth", "notional")
BOOK_DEPTH_BANDS = (0.2, 1.0, 5.0)


@dataclass(frozen=True)
class BinanceBookDepthConfig:
    output_root: Path = Path("data/external/binance_um_book_depth")
    download_workers: int = 16
    timeout_seconds: int = 90
    retry_attempts: int = 4
    retry_sleep_seconds: float = 1.0
    mapping_probe_day: date = date(2025, 8, 4)
    retain_raw_archives: bool = True


@dataclass(frozen=True)
class BinanceBookDepthSymbolResult:
    bybit_symbol: str
    binance_symbol: str | None
    requested_days: int
    downloaded_days: int
    cached_days: int
    missing_days: int
    invalid_days: int
    feature_rows: int
    first_day: str | None
    last_day: str | None
    output_path: str | None
    error: str | None = None


def book_depth_daily_url(binance_symbol: str, day: date) -> str:
    stamp = day.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/daily/bookDepth/"
        f"{binance_symbol}/{binance_symbol}-bookDepth-{stamp}.zip"
    )


def _request_bytes(
    url: str,
    cfg: BinanceBookDepthConfig,
    *,
    method: str = "GET",
    client: httpx.Client | None = None,
) -> bytes | None:
    owns_client = client is None
    session = client or httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    )
    try:
        for attempt in range(cfg.retry_attempts):
            try:
                response = session.request(method, url)
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
    finally:
        if owns_client:
            session.close()


def resolve_binance_book_depth_symbol(
    bybit_symbol: str,
    cfg: BinanceBookDepthConfig = BinanceBookDepthConfig(),
    client: httpx.Client | None = None,
) -> str | None:
    """Resolve exchange multiplier aliases without reading strategy outcomes."""
    for candidate in binance_symbol_candidates(bybit_symbol):
        if (
            _request_bytes(
                book_depth_daily_url(candidate, cfg.mapping_probe_day),
                cfg,
                method="HEAD",
                client=client,
            )
            is not None
        ):
            return candidate
    return None


def _band_label(band: float) -> str:
    return str(band).replace(".", "p")


def parse_book_depth_zip(
    content: bytes,
    bybit_symbol: str,
    source_day: date,
) -> dict[str, object]:
    """Validate one archive and reduce it to frozen daily imbalance statistics."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in book-depth archive, found {csv_names!r}")
        frame = pd.read_csv(archive.open(csv_names[0]))
    missing = set(BOOK_DEPTH_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Book-depth archive is missing columns: {sorted(missing)}")
    frame = frame.loc[:, BOOK_DEPTH_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("percentage", "depth", "notional"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(BOOK_DEPTH_COLUMNS))
    expected_day = pd.Timestamp(source_day, tz="UTC")
    frame = frame[frame["timestamp"].dt.floor("D").eq(expected_day)]
    if frame.empty:
        raise ValueError(f"No valid snapshots for source day {source_day.isoformat()}")

    row: dict[str, object] = {
        "bybit_symbol": bybit_symbol,
        "source_day": expected_day,
        "snapshot_count": int(frame["timestamp"].nunique()),
        "first_snapshot": frame["timestamp"].min(),
        "last_snapshot": frame["timestamp"].max(),
    }
    for band in BOOK_DEPTH_BANDS:
        label = _band_label(band)
        local = frame[np.isclose(frame["percentage"].abs(), band)].copy()
        for value_column in ("depth", "notional"):
            pivot = local.pivot_table(
                index="timestamp",
                columns="percentage",
                values=value_column,
                aggfunc="last",
                observed=True,
            )
            negative = next(
                (column for column in pivot.columns if np.isclose(column, -band)), None
            )
            positive = next(
                (column for column in pivot.columns if np.isclose(column, band)), None
            )
            if negative is None or positive is None:
                imbalance = pd.Series(dtype=float)
            else:
                denominator = pivot[negative] + pivot[positive]
                imbalance = ((pivot[negative] - pivot[positive]) / denominator).where(
                    denominator.gt(0)
                )
                imbalance = imbalance.replace([np.inf, -np.inf], np.nan).dropna()
            prefix = f"{value_column}_imbalance_{label}"
            row[f"{prefix}_median"] = float(imbalance.median())
            row[f"{prefix}_mean"] = float(imbalance.mean())
            row[f"{prefix}_std"] = float(imbalance.std(ddof=0))
            row[f"{prefix}_valid_snapshots"] = int(imbalance.count())
    return row


def _days(start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        raise ValueError("end_day must not precede start_day")
    return [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]


def _atomic_write_bytes(content: bytes, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _archive_path(cfg: BinanceBookDepthConfig, symbol: str, day: date) -> Path:
    return (
        cfg.output_root
        / "raw"
        / symbol
        / day.strftime("%Y-%m")
        / f"{symbol}-bookDepth-{day.isoformat()}.zip"
    )


def download_book_depth_symbol(
    bybit_symbol: str,
    start_day: date,
    end_day: date,
    cfg: BinanceBookDepthConfig = BinanceBookDepthConfig(),
) -> tuple[BinanceBookDepthSymbolResult, pd.DataFrame]:
    """Download, cache and aggregate one fixed symbol over an inclusive day range."""
    requested_days = _days(start_day, end_day)
    features: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    downloaded_days = 0
    cached_days = 0
    missing_days = 0
    invalid_days = 0
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        binance_symbol = resolve_binance_book_depth_symbol(bybit_symbol, cfg, client)
        if binance_symbol is None:
            result = BinanceBookDepthSymbolResult(
                bybit_symbol,
                None,
                len(requested_days),
                0,
                0,
                len(requested_days),
                0,
                0,
                None,
                None,
                None,
                "no_binance_book_depth_symbol",
            )
            return result, pd.DataFrame()
        for day in requested_days:
            path = _archive_path(cfg, binance_symbol, day)
            cached = path.exists() and path.stat().st_size > 0
            content = path.read_bytes() if cached else None
            error: str | None = None
            if content is None:
                try:
                    content = _request_bytes(
                        book_depth_daily_url(binance_symbol, day), cfg, client=client
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                if content is not None:
                    downloaded_days += 1
                    if cfg.retain_raw_archives:
                        _atomic_write_bytes(content, path)
            else:
                cached_days += 1
            if content is None:
                missing_days += 1
                manifest_rows.append(
                    {
                        "bybit_symbol": bybit_symbol,
                        "binance_symbol": binance_symbol,
                        "source_day": day.isoformat(),
                        "status": "error" if error else "missing",
                        "cached": False,
                        "bytes": 0,
                        "sha256": None,
                        "error": error,
                    }
                )
                continue
            try:
                feature = parse_book_depth_zip(content, bybit_symbol, day)
                feature["binance_symbol"] = binance_symbol
                feature["archive_sha256"] = hashlib.sha256(content).hexdigest()
                features.append(feature)
                status = "ok"
            except Exception as exc:
                invalid_days += 1
                status = "invalid"
                error = f"{type(exc).__name__}: {exc}"
            manifest_rows.append(
                {
                    "bybit_symbol": bybit_symbol,
                    "binance_symbol": binance_symbol,
                    "source_day": day.isoformat(),
                    "status": status,
                    "cached": cached,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "error": error,
                }
            )

    feature_frame = pd.DataFrame(features)
    output_path: Path | None = None
    first_day: str | None = None
    last_day: str | None = None
    if not feature_frame.empty:
        feature_frame = feature_frame.sort_values("source_day").reset_index(drop=True)
        output_path = cfg.output_root / "daily_features" / f"{bybit_symbol}.parquet"
        _atomic_write_parquet(feature_frame, output_path)
        first_day = pd.Timestamp(feature_frame["source_day"].min()).date().isoformat()
        last_day = pd.Timestamp(feature_frame["source_day"].max()).date().isoformat()
    result = BinanceBookDepthSymbolResult(
        bybit_symbol=bybit_symbol,
        binance_symbol=binance_symbol,
        requested_days=len(requested_days),
        downloaded_days=downloaded_days,
        cached_days=cached_days,
        missing_days=missing_days,
        invalid_days=invalid_days,
        feature_rows=len(feature_frame),
        first_day=first_day,
        last_day=last_day,
        output_path=str(output_path) if output_path else None,
        error=None if len(feature_frame) else "no_valid_daily_archives",
    )
    return result, pd.DataFrame(manifest_rows)


def backfill_binance_book_depth(
    symbols: list[str],
    start_day: date,
    end_day: date,
    cfg: BinanceBookDepthConfig = BinanceBookDepthConfig(),
) -> pd.DataFrame:
    """Run symbol downloads concurrently and persist audit-grade coverage metadata."""
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    results: list[BinanceBookDepthSymbolResult] = []
    daily_manifests: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {
            executor.submit(download_book_depth_symbol, symbol, start_day, end_day, cfg): symbol
            for symbol in normalized
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result, daily = future.result()
            except Exception as exc:
                result = BinanceBookDepthSymbolResult(
                    symbol,
                    None,
                    len(_days(start_day, end_day)),
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    f"{type(exc).__name__}: {exc}",
                )
                daily = pd.DataFrame()
            results.append(result)
            if not daily.empty:
                daily_manifests.append(daily)
            print(
                f"bookDepth: {completed}/{len(normalized)} {symbol} "
                f"days={result.feature_rows} missing={result.missing_days} "
                f"invalid={result.invalid_days} error={result.error or '-'}",
                flush=True,
            )

    root = ensure_dir(cfg.output_root)
    manifest = pd.DataFrame([asdict(result) for result in results]).sort_values("bybit_symbol")
    manifest.to_csv(root / "manifest.csv", index=False)
    daily_manifest = (
        pd.concat(daily_manifests, ignore_index=True)
        if daily_manifests
        else pd.DataFrame()
    )
    if not daily_manifest.empty:
        _atomic_write_parquet(
            daily_manifest.sort_values(["source_day", "bybit_symbol"]),
            root / "daily_manifest.parquet",
        )
    coverage = {
        "start_day": start_day.isoformat(),
        "end_day": end_day.isoformat(),
        "requested_symbols": len(normalized),
        "covered_symbols": int(manifest["feature_rows"].gt(0).sum()),
        "feature_rows": int(manifest["feature_rows"].sum()),
        "raw_bytes": int(daily_manifest["bytes"].sum()) if not daily_manifest.empty else 0,
        "config": {
            **asdict(cfg),
            "output_root": str(cfg.output_root),
            "mapping_probe_day": cfg.mapping_probe_day.isoformat(),
        },
    }
    (root / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return manifest.reset_index(drop=True)
