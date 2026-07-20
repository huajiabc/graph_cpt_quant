"""Historical Binance USD-M positioning metrics from public daily archives."""

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
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.orderflow_history import binance_symbol_candidates


BINANCE_PUBLIC_DATA_BASE = "https://data.binance.vision/data"
METRICS_COLUMNS = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
NUMERIC_COLUMNS = METRICS_COLUMNS[2:]


@dataclass(frozen=True)
class BinanceMetricsConfig:
    output_root: Path = Path("data/external/binance_um_metrics_5m")
    download_workers: int = 32
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_sleep_seconds: float = 1.0
    mapping_probe_day: date = date(2026, 6, 1)
    merge_existing: bool = True


@dataclass(frozen=True)
class BinanceMetricsSymbolResult:
    bybit_symbol: str
    binance_symbol: str | None
    requested_days: int
    downloaded_days: int
    missing_days: int
    rows: int
    first_time: str | None
    last_time: str | None
    output_path: str | None
    error: str | None = None


def metrics_daily_url(binance_symbol: str, day: date) -> str:
    stamp = day.strftime("%Y-%m-%d")
    return (
        f"{BINANCE_PUBLIC_DATA_BASE}/futures/um/daily/metrics/"
        f"{binance_symbol}/{binance_symbol}-metrics-{stamp}.zip"
    )


def _request_bytes(
    url: str,
    cfg: BinanceMetricsConfig,
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


def resolve_binance_metrics_symbol(
    bybit_symbol: str,
    cfg: BinanceMetricsConfig = BinanceMetricsConfig(),
    client: httpx.Client | None = None,
) -> str | None:
    """Resolve multiplier aliases with one recent archive probe per candidate."""
    for candidate in binance_symbol_candidates(bybit_symbol):
        if (
            _request_bytes(
                metrics_daily_url(candidate, cfg.mapping_probe_day),
                cfg,
                method="HEAD",
                client=client,
            )
            is not None
        ):
            return candidate
    return None


def parse_metrics_zip(content: bytes, bybit_symbol: str) -> pd.DataFrame:
    """Parse and validate one daily public metrics archive."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in metrics archive, found {csv_names!r}")
        frame = pd.read_csv(archive.open(csv_names[0]))
    missing = set(METRICS_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Metrics archive is missing columns: {sorted(missing)}")
    frame = frame.loc[:, METRICS_COLUMNS].copy()
    frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["binance_symbol"] = frame["symbol"].astype(str)
    frame["bybit_symbol"] = bybit_symbol
    frame = frame.drop(columns="symbol")
    return frame.dropna(subset=["create_time"]).reset_index(drop=True)


def _days(start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        raise ValueError("end_day must not precede start_day")
    return [start_day + timedelta(days=offset) for offset in range((end_day - start_day).days + 1)]


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _merge_existing_metrics(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return frame
    existing = pd.read_parquet(path)
    return (
        pd.concat([existing, frame], ignore_index=True)
        .drop_duplicates(["bybit_symbol", "create_time"], keep="last")
        .sort_values("create_time")
        .reset_index(drop=True)
    )


def inventory_binance_metrics(
    output_root: Path,
    expected_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a full-directory inventory without relying on the last download call."""
    rows: list[dict[str, object]] = []
    covered: set[str] = set()
    for path in sorted(output_root.glob("*.parquet")):
        symbol = path.stem.upper()
        frame = pd.read_parquet(
            path,
            columns=["create_time", "binance_symbol", "bybit_symbol", "source_day"],
        )
        frame["create_time"] = pd.to_datetime(
            frame["create_time"], utc=True, errors="coerce"
        )
        frame["source_day"] = pd.to_datetime(frame["source_day"], errors="coerce")
        valid_time = frame["create_time"].dropna()
        valid_days = frame["source_day"].dropna().dt.normalize()
        first_day = valid_days.min() if not valid_days.empty else pd.NaT
        last_day = valid_days.max() if not valid_days.empty else pd.NaT
        requested_days = (
            int((last_day - first_day).days + 1)
            if pd.notna(first_day) and pd.notna(last_day)
            else 0
        )
        downloaded_days = int(valid_days.nunique())
        binance_values = frame["binance_symbol"].dropna().astype(str)
        bybit_values = frame["bybit_symbol"].dropna().astype(str)
        rows.append(
            {
                "bybit_symbol": bybit_values.iloc[0] if len(bybit_values) else symbol,
                "binance_symbol": (
                    binance_values.iloc[0] if len(binance_values) else None
                ),
                "requested_days": requested_days,
                "downloaded_days": downloaded_days,
                "missing_days": max(requested_days - downloaded_days, 0),
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
    expected = sorted(
        {
            symbol.upper().strip()
            for symbol in (expected_symbols or [])
            if symbol.strip()
        }
    )
    for symbol in expected:
        if symbol not in covered:
            rows.append(
                {
                    "bybit_symbol": symbol,
                    "binance_symbol": None,
                    "requested_days": 0,
                    "downloaded_days": 0,
                    "missing_days": 0,
                    "rows": 0,
                    "first_time": None,
                    "last_time": None,
                    "output_path": None,
                    "error": "missing_data_file",
                }
            )
    return pd.DataFrame(rows).sort_values("bybit_symbol").reset_index(drop=True)


def write_binance_metrics_inventory(
    cfg: BinanceMetricsConfig,
    expected_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Write a consolidated manifest and coverage record for every local file."""
    manifest = inventory_binance_metrics(cfg.output_root, expected_symbols)
    ensure_dir(cfg.output_root)
    manifest.to_csv(cfg.output_root / "manifest.csv", index=False)
    covered = manifest[manifest["rows"].gt(0)]
    first_times = pd.to_datetime(covered["first_time"], utc=True, errors="coerce")
    last_times = pd.to_datetime(covered["last_time"], utc=True, errors="coerce")
    coverage = {
        "start_day": (
            first_times.min().date().isoformat() if first_times.notna().any() else None
        ),
        "end_day": (
            last_times.max().date().isoformat() if last_times.notna().any() else None
        ),
        "requested_symbols": len(manifest),
        "covered_symbols": int(manifest["rows"].gt(0).sum()),
        "rows": int(manifest["rows"].sum()),
        "config": {
            **asdict(cfg),
            "output_root": str(cfg.output_root),
            "mapping_probe_day": cfg.mapping_probe_day.isoformat(),
            "inventory_mode": "full_directory",
        },
    }
    (cfg.output_root / "coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    return manifest


def download_metrics_symbol(
    bybit_symbol: str,
    start_day: date,
    end_day: date,
    cfg: BinanceMetricsConfig = BinanceMetricsConfig(),
) -> BinanceMetricsSymbolResult:
    requested_days = _days(start_day, end_day)
    frames: list[pd.DataFrame] = []
    missing_days = 0
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        binance_symbol = resolve_binance_metrics_symbol(bybit_symbol, cfg, client)
        if binance_symbol is None:
            return BinanceMetricsSymbolResult(
                bybit_symbol=bybit_symbol,
                binance_symbol=None,
                requested_days=len(requested_days),
                downloaded_days=0,
                missing_days=len(requested_days),
                rows=0,
                first_time=None,
                last_time=None,
                output_path=None,
                error="no_binance_metrics_symbol",
            )
        for day in requested_days:
            content = _request_bytes(metrics_daily_url(binance_symbol, day), cfg, client=client)
            if content is None:
                missing_days += 1
                continue
            daily = parse_metrics_zip(content, bybit_symbol)
            daily["source_day"] = pd.Timestamp(day)
            frames.append(daily)
    if not frames:
        return BinanceMetricsSymbolResult(
            bybit_symbol=bybit_symbol,
            binance_symbol=binance_symbol,
            requested_days=len(requested_days),
            downloaded_days=0,
            missing_days=missing_days,
            rows=0,
            first_time=None,
            last_time=None,
            output_path=None,
            error="no_daily_archives",
        )

    frame = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["bybit_symbol", "create_time"], keep="last")
        .sort_values("create_time")
        .reset_index(drop=True)
    )
    output_path = cfg.output_root / f"{bybit_symbol}.parquet"
    if cfg.merge_existing:
        frame = _merge_existing_metrics(frame, output_path)
    _atomic_write_parquet(frame, output_path)
    return BinanceMetricsSymbolResult(
        bybit_symbol=bybit_symbol,
        binance_symbol=binance_symbol,
        requested_days=len(requested_days),
        downloaded_days=len(frames),
        missing_days=missing_days,
        rows=len(frame),
        first_time=frame["create_time"].min().isoformat(),
        last_time=frame["create_time"].max().isoformat(),
        output_path=str(output_path),
    )


def backfill_binance_metrics(
    symbols: list[str],
    start_day: date,
    end_day: date,
    cfg: BinanceMetricsConfig = BinanceMetricsConfig(),
) -> pd.DataFrame:
    """Download symbols concurrently and write a reproducibility manifest."""
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    existing_manifest_path = cfg.output_root / "manifest.csv"
    previous_symbols: set[str] = set()
    if existing_manifest_path.exists():
        previous = pd.read_csv(existing_manifest_path, usecols=["bybit_symbol"])
        previous_symbols = set(previous["bybit_symbol"].dropna().astype(str).str.upper())
    results: list[BinanceMetricsSymbolResult] = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {
            executor.submit(download_metrics_symbol, symbol, start_day, end_day, cfg): symbol
            for symbol in normalized
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # manifest the failure; other symbols remain usable
                result = BinanceMetricsSymbolResult(
                    bybit_symbol=symbol,
                    binance_symbol=None,
                    requested_days=len(_days(start_day, end_day)),
                    downloaded_days=0,
                    missing_days=0,
                    rows=0,
                    first_time=None,
                    last_time=None,
                    output_path=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            print(
                f"metrics: {completed}/{len(normalized)} {symbol} "
                f"days={result.downloaded_days} rows={result.rows} "
                f"error={result.error or '-'}",
                flush=True,
            )

    request_manifest = pd.DataFrame([asdict(result) for result in results]).sort_values(
        "bybit_symbol"
    )
    write_binance_metrics_inventory(
        cfg,
        expected_symbols=sorted(previous_symbols | set(normalized)),
    )
    return request_manifest.reset_index(drop=True)
