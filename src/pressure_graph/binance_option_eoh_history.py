"""Official Binance option EOH summaries and matching USD-M hourly prices."""

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
import pandas as pd

from pressure_graph.io import ensure_dir


BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data"
OPTION_EOH_COLUMNS = (
    "date",
    "hour",
    "symbol",
    "underlying",
    "type",
    "strike",
    "open",
    "high",
    "low",
    "close",
    "volume_contracts",
    "volume_usdt",
    "best_bid_price",
    "best_ask_price",
    "best_bid_qty",
    "best_ask_qty",
    "best_buy_iv",
    "best_sell_iv",
    "mark_price",
    "mark_iv",
    "delta",
    "gamma",
    "vega",
    "theta",
    "openinterest_contracts",
    "openinterest_usdt",
)
OPTION_NUMERIC_COLUMNS = tuple(
    column
    for column in OPTION_EOH_COLUMNS
    if column not in {"date", "symbol", "underlying", "type", "strike"}
)
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)


@dataclass(frozen=True)
class BinanceOptionHistoryConfig:
    output_root: Path = Path("data/external/binance_option_vol_front")
    download_workers: int = 12
    timeout_seconds: int = 90
    retry_attempts: int = 4
    retry_sleep_seconds: float = 1.0
    retained_option_hours: tuple[int, ...] = (0,)
    retain_raw_archives: bool = True


def option_eoh_daily_url(underlying: str, day: date) -> str:
    stamp = day.isoformat()
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/option/daily/EOHSummary/{underlying}/"
        f"{underlying}-EOHSummary-{stamp}.zip"
    )


def um_hourly_month_url(symbol: str, month: str) -> str:
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/monthly/klines/{symbol}/1h/"
        f"{symbol}-1h-{month}.zip"
    )


def _request_bytes(
    url: str,
    cfg: BinanceOptionHistoryConfig,
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
                response = session.get(url)
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


def _single_csv_frame(content: bytes, **read_csv_kwargs: object) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in archive, found {names!r}")
        return pd.read_csv(archive.open(names[0]), **read_csv_kwargs)


def parse_option_eoh_zip(
    content: bytes,
    source_day: date,
    retained_hours: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """Parse selected EOH hours and apply a conservative end-of-hour timestamp."""
    frame = _single_csv_frame(content)
    missing = set(OPTION_EOH_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Option EOH archive is missing columns: {sorted(missing)}")
    frame = frame.loc[:, OPTION_EOH_COLUMNS].copy()
    for column in OPTION_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["archive_date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame[frame["hour"].isin(retained_hours)].copy()
    frame["snapshot_time"] = frame["archive_date"] + pd.to_timedelta(
        frame["hour"] + 1, unit="h"
    )

    parsed = frame["symbol"].astype(str).str.extract(
        r"^[A-Z]+-(?P<expiry_code>\d{6})-(?P<strike_price>\d+(?:\.\d+)?)-"
        r"(?P<option_type>[CP])$"
    )
    frame["expiration_time"] = pd.to_datetime(
        parsed["expiry_code"], format="%y%m%d", utc=True, errors="coerce"
    ) + pd.Timedelta(hours=8)
    frame["strike_price"] = pd.to_numeric(parsed["strike_price"], errors="coerce")
    frame["option_type"] = parsed["option_type"]

    expected_day = pd.Timestamp(source_day, tz="UTC")
    frame = frame[
        frame["archive_date"].eq(expected_day)
        & frame["snapshot_time"].notna()
        & frame["expiration_time"].notna()
        & frame["strike_price"].gt(0)
        & frame["option_type"].isin(["C", "P"])
    ].copy()
    if frame.empty:
        raise ValueError(f"No retained valid option rows for {source_day.isoformat()}")
    frame["source_day"] = expected_day
    return (
        frame.drop(columns=["date", "strike", "type"])
        .drop_duplicates(["snapshot_time", "symbol"], keep="last")
        .sort_values(["snapshot_time", "expiration_time", "strike_price", "option_type"])
        .reset_index(drop=True)
    )


def parse_um_hourly_kline_zip(content: bytes, symbol: str) -> pd.DataFrame:
    """Parse one official Binance USD-M monthly one-hour kline archive."""
    frame = _single_csv_frame(content, header=None)
    if frame.shape[1] < len(KLINE_COLUMNS):
        raise ValueError(f"Kline archive has only {frame.shape[1]} columns")
    frame = frame.iloc[:, : len(KLINE_COLUMNS)].copy()
    frame.columns = KLINE_COLUMNS
    for column in KLINE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bar_open_time"] = pd.to_datetime(
        frame["open_time"], unit="ms", utc=True, errors="coerce"
    )
    frame["bar_close_time"] = pd.to_datetime(
        frame["close_time"], unit="ms", utc=True, errors="coerce"
    )
    frame = frame.dropna(
        subset=["bar_open_time", "bar_close_time", "open", "high", "low", "close"]
    ).copy()
    if frame.empty:
        raise ValueError(f"No valid hourly klines for {symbol}")
    frame["symbol"] = symbol
    return (
        frame[
            [
                "symbol",
                "bar_open_time",
                "bar_close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades",
            ]
        ]
        .drop_duplicates("bar_open_time", keep="last")
        .sort_values("bar_open_time")
        .reset_index(drop=True)
    )


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


def _days(start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        raise ValueError("end_day must not precede start_day")
    return [
        start_day + timedelta(days=offset)
        for offset in range((end_day - start_day).days + 1)
    ]


def _months(start_day: date, end_day: date) -> list[str]:
    start = pd.Timestamp(start_day).to_period("M")
    end = pd.Timestamp(end_day).to_period("M")
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _download_cached(
    url: str,
    path: Path,
    cfg: BinanceOptionHistoryConfig,
) -> tuple[bytes | None, bool, str | None]:
    cached = path.exists() and path.stat().st_size > 0
    if cached:
        return path.read_bytes(), True, None
    try:
        content = _request_bytes(url, cfg)
    except Exception as exc:
        return None, False, f"{type(exc).__name__}: {exc}"
    if content is not None and cfg.retain_raw_archives:
        _atomic_write_bytes(content, path)
    return content, False, None


def _download_option_day(
    underlying: str,
    source_day: date,
    cfg: BinanceOptionHistoryConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = (
        cfg.output_root
        / "raw_option_eoh"
        / underlying
        / source_day.strftime("%Y-%m")
        / f"{underlying}-EOHSummary-{source_day.isoformat()}.zip"
    )
    content, cached, error = _download_cached(
        option_eoh_daily_url(underlying, source_day), path, cfg
    )
    if content is None:
        return pd.DataFrame(), {
            "asset": underlying,
            "period": source_day.isoformat(),
            "kind": "option_eoh",
            "status": "error" if error else "missing",
            "cached": cached,
            "bytes": 0,
            "sha256": None,
            "rows": 0,
            "error": error,
        }
    try:
        frame = parse_option_eoh_zip(content, source_day, cfg.retained_option_hours)
        frame["archive_sha256"] = hashlib.sha256(content).hexdigest()
        status = "ok"
    except Exception as exc:
        frame = pd.DataFrame()
        status = "invalid"
        error = f"{type(exc).__name__}: {exc}"
    return frame, {
        "asset": underlying,
        "period": source_day.isoformat(),
        "kind": "option_eoh",
        "status": status,
        "cached": cached,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(frame),
        "error": error,
    }


def _download_kline_month(
    symbol: str,
    month: str,
    cfg: BinanceOptionHistoryConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = (
        cfg.output_root
        / "raw_um_klines_1h"
        / symbol
        / f"{symbol}-1h-{month}.zip"
    )
    content, cached, error = _download_cached(um_hourly_month_url(symbol, month), path, cfg)
    if content is None:
        return pd.DataFrame(), {
            "asset": symbol,
            "period": month,
            "kind": "um_kline_1h",
            "status": "error" if error else "missing",
            "cached": cached,
            "bytes": 0,
            "sha256": None,
            "rows": 0,
            "error": error,
        }
    try:
        frame = parse_um_hourly_kline_zip(content, symbol)
        status = "ok"
    except Exception as exc:
        frame = pd.DataFrame()
        status = "invalid"
        error = f"{type(exc).__name__}: {exc}"
    return frame, {
        "asset": symbol,
        "period": month,
        "kind": "um_kline_1h",
        "status": status,
        "cached": cached,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(frame),
        "error": error,
    }


def backfill_binance_option_vol_front(
    underlying: str,
    symbols: list[str],
    start_day: date,
    end_day: date,
    cfg: BinanceOptionHistoryConfig = BinanceOptionHistoryConfig(),
) -> dict[str, Path]:
    """Download auditable option snapshots and matching perpetual hourly prices."""
    normalized_symbols = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    option_frames: list[pd.DataFrame] = []
    kline_frames: dict[str, list[pd.DataFrame]] = {
        symbol: [] for symbol in normalized_symbols
    }
    manifest_rows: list[dict[str, object]] = []
    option_days = _days(start_day, end_day)
    months = _months(start_day - timedelta(days=31), end_day + timedelta(days=1))

    jobs: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {}
        for source_day in option_days:
            future = executor.submit(_download_option_day, underlying, source_day, cfg)
            futures[future] = ("option", source_day.isoformat())
        for symbol in normalized_symbols:
            for month in months:
                future = executor.submit(_download_kline_month, symbol, month, cfg)
                futures[future] = (symbol, month)
        for completed, future in enumerate(as_completed(futures), start=1):
            key, period = futures[future]
            frame, manifest = future.result()
            manifest_rows.append(manifest)
            if key == "option":
                if not frame.empty:
                    option_frames.append(frame)
            elif not frame.empty:
                kline_frames[key].append(frame)
            jobs.append((key, period))
            if completed % 25 == 0 or completed == len(futures):
                print(f"option-vol-front download: {completed}/{len(futures)}", flush=True)

    root = ensure_dir(cfg.output_root)
    option_output = root / "option_eoh_hour0.parquet"
    if not option_frames:
        raise RuntimeError("No valid Binance option EOH archives were downloaded")
    options = (
        pd.concat(option_frames, ignore_index=True)
        .drop_duplicates(["snapshot_time", "symbol"], keep="last")
        .sort_values(["snapshot_time", "expiration_time", "strike_price", "option_type"])
        .reset_index(drop=True)
    )
    _atomic_write_parquet(options, option_output)

    price_root = ensure_dir(root / "um_klines_1h")
    covered_symbols = 0
    for symbol, frames in kline_frames.items():
        if not frames:
            continue
        covered_symbols += 1
        hourly = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("bar_open_time", keep="last")
            .sort_values("bar_open_time")
            .reset_index(drop=True)
        )
        _atomic_write_parquet(hourly, price_root / f"{symbol}.parquet")

    manifest = pd.DataFrame(manifest_rows).sort_values(["kind", "asset", "period"])
    manifest_path = root / "manifest.parquet"
    _atomic_write_parquet(manifest, manifest_path)
    coverage = {
        "underlying": underlying,
        "start_day": start_day.isoformat(),
        "end_day": end_day.isoformat(),
        "requested_option_days": len(option_days),
        "covered_option_days": int(options["source_day"].nunique()),
        "retained_option_rows": len(options),
        "requested_price_symbols": len(normalized_symbols),
        "covered_price_symbols": covered_symbols,
        "download_jobs": len(jobs),
        "config": {
            **asdict(cfg),
            "output_root": str(cfg.output_root),
            "retained_option_hours": list(cfg.retained_option_hours),
        },
    }
    coverage_path = root / "coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return {
        "options": option_output,
        "prices": price_root,
        "manifest": manifest_path,
        "coverage": coverage_path,
    }
