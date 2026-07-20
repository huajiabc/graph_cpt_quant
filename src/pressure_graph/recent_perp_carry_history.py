"""Recent Bybit/Binance perpetual data for forward carry extension."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pandas as pd

from pressure_graph.binance_um_carry_history import (
    BinanceUmCarryConfig,
    fetch_um_funding,
    um_symbol,
)
from pressure_graph.clients.bybit import BybitClient
from pressure_graph.io import ensure_dir


BINANCE_FAPI_BASE = "https://fapi.binance.com"


@dataclass(frozen=True)
class RecentPerpCarryConfig:
    output_root: Path = Path("data/external/recent_perp_carry")
    download_workers: int = 6
    timeout_seconds: int = 60


@dataclass(frozen=True)
class RecentPerpCarryResult:
    symbol: str
    bybit_kline_rows: int
    bybit_funding_rows: int
    binance_kline_rows: int
    binance_funding_rows: int
    error: str | None = None


def parse_binance_um_klines(
    rows: list[list[object]], bybit_symbol: str
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["bar_open_time"] = pd.to_datetime(
        pd.to_numeric(frame["open_time"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    frame["feature_time"] = frame["bar_open_time"] + pd.Timedelta(hours=1)
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bybit_symbol"] = bybit_symbol
    frame["binance_symbol"] = um_symbol(bybit_symbol)
    wanted = [
        "bybit_symbol",
        "binance_symbol",
        "bar_open_time",
        "feature_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    return (
        frame[wanted]
        .dropna(subset=["bar_open_time", "feature_time", "close"])
        .drop_duplicates(["bybit_symbol", "bar_open_time"], keep="last")
        .sort_values("bar_open_time")
        .reset_index(drop=True)
    )


def fetch_binance_um_klines(
    bybit_symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: httpx.Client,
) -> pd.DataFrame:
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) - 1
    rows: list[list[object]] = []
    while cursor <= end_ms:
        response = client.get(
            f"{BINANCE_FAPI_BASE}/fapi/v1/klines",
            params={
                "symbol": um_symbol(bybit_symbol),
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if response.status_code == 400 and response.json().get("code") == -1121:
            return pd.DataFrame()
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        latest = max(int(row[0]) for row in batch)
        if len(batch) < 1500:
            break
        cursor = latest + 60 * 60 * 1000
    return parse_binance_um_klines(rows, bybit_symbol)


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def download_recent_perp_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: RecentPerpCarryConfig = RecentPerpCarryConfig(),
) -> RecentPerpCarryResult:
    errors = []
    bybit = BybitClient()
    try:
        bybit_klines = bybit.klines(symbol, start, end, "15m")
        bybit_funding = bybit.funding_history(symbol, start, end)
    except Exception as exc:
        bybit_klines = pd.DataFrame()
        bybit_funding = pd.DataFrame()
        errors.append(f"bybit:{type(exc).__name__}:{exc}")
    finally:
        bybit.close()
    with httpx.Client(
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        try:
            binance_klines = fetch_binance_um_klines(symbol, start, end, client)
            binance_funding = fetch_um_funding(
                symbol,
                start.date(),
                (end - pd.Timedelta(days=1)).date(),
                client,
                BinanceUmCarryConfig(timeout_seconds=cfg.timeout_seconds),
            )
        except Exception as exc:
            binance_klines = pd.DataFrame()
            binance_funding = pd.DataFrame()
            errors.append(f"binance:{type(exc).__name__}:{exc}")
    for name, frame in (
        ("bybit_klines_15m", bybit_klines),
        ("bybit_funding", bybit_funding),
        ("binance_klines_1h", binance_klines),
        ("binance_funding", binance_funding),
    ):
        if not frame.empty:
            _atomic_write(frame, cfg.output_root / name / f"{symbol}.parquet")
    return RecentPerpCarryResult(
        symbol=symbol,
        bybit_kline_rows=len(bybit_klines),
        bybit_funding_rows=len(bybit_funding),
        binance_kline_rows=len(binance_klines),
        binance_funding_rows=len(binance_funding),
        error="|".join(errors) or None,
    )


def backfill_recent_perp_carry(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: RecentPerpCarryConfig = RecentPerpCarryConfig(),
) -> pd.DataFrame:
    normalized = sorted({symbol.upper().strip() for symbol in symbols if symbol.strip()})
    results = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as executor:
        futures = {
            executor.submit(download_recent_perp_symbol, symbol, start, end, cfg): symbol
            for symbol in normalized
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = RecentPerpCarryResult(
                    symbol=symbol,
                    bybit_kline_rows=0,
                    bybit_funding_rows=0,
                    binance_kline_rows=0,
                    binance_funding_rows=0,
                    error=f"{type(exc).__name__}:{exc}",
                )
            results.append(result)
            print(
                f"recent: {completed}/{len(normalized)} {symbol} "
                f"bybit={result.bybit_kline_rows}/{result.bybit_funding_rows} "
                f"binance={result.binance_kline_rows}/{result.binance_funding_rows} "
                f"error={result.error or '-'}",
                flush=True,
            )
    manifest = pd.DataFrame([asdict(result) for result in results]).sort_values("symbol")
    ensure_dir(cfg.output_root)
    manifest.to_csv(cfg.output_root / "manifest.csv", index=False)
    (cfg.output_root / "coverage.json").write_text(
        json.dumps(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "symbols": len(normalized),
                "complete_bybit": int(
                    (manifest["bybit_kline_rows"].gt(0) & manifest["bybit_funding_rows"].gt(0)).sum()
                ),
                "complete_binance": int(
                    (manifest["binance_kline_rows"].gt(0) & manifest["binance_funding_rows"].gt(0)).sum()
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest.reset_index(drop=True)
