"""Bybit linear-perp continuous CVD backfill.

Mirrors ``binance_continuous_cvd.py``: per (symbol, day) it downloads the
free public trade dump from ``public.bybit.com``, bins trades into fixed-size
bars, and writes per-month parquets with the SAME schema the Binance writer
produces. The Sell-Pressure Propagation Map (Phase 2B) consumes both with no
exchange-specific code paths.

Schema parity columns (compatible with ``binance_um/continuous``):
    symbol, bar_open_time, bar_size, trade_count, volume, turnover,
    buy_volume, sell_volume, buy_turnover, sell_turnover,
    taker_buy_ratio, buy_sell_imbalance, cvd_delta_volume, cvd_delta_turnover,
    large_trade_threshold, large_buy_count, large_sell_count,
    large_buy_turnover, large_sell_turnover, coverage_ratio, source_quality

Source: Bybit publishes one gzip CSV per symbol-day at
    https://public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz

Raw row schema (subset used): timestamp, side ('Buy'|'Sell'), size,
homeNotional, foreignNotional (= turnover in quote / USDT).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import gzip
import io
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

# Mirror of binance_continuous_cvd layout: data/orderflow_history/<venue>/...
DEFAULT_HISTORY_ROOT = Path("data/orderflow_history/bybit_linear")
DEFAULT_BAR_SIZES: tuple[str, ...] = ("1min", "5min", "15min")
LARGE_TRADE_TURNOVER_QUANTILE: float = 0.95
PUBLIC_URL_BASE = "https://public.bybit.com/trading"


@dataclass(frozen=True)
class BybitCvdConfig:
    history_root: Path = DEFAULT_HISTORY_ROOT
    bar_sizes: tuple[str, ...] = DEFAULT_BAR_SIZES
    large_trade_turnover_quantile: float = LARGE_TRADE_TURNOVER_QUANTILE
    timeout_seconds: int = 120


def raw_path(cfg: BybitCvdConfig, symbol: str, day: date) -> Path:
    return cfg.history_root / "raw" / "trades" / symbol / f"{symbol}{day.isoformat()}.csv.gz"


def continuous_path(cfg: BybitCvdConfig, symbol: str, bar_size: str, month: str) -> Path:
    return cfg.history_root / "continuous" / symbol / bar_size / f"{month}.parquet"


def download_bybit_day(cfg: BybitCvdConfig, symbol: str, day: date, *, force: bool = False) -> Path:
    """Download one (symbol, day) gzip CSV from public.bybit.com if not cached."""
    out = raw_path(cfg, symbol, day)
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PUBLIC_URL_BASE}/{symbol}/{symbol}{day.isoformat()}.csv.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "pressure-graph/0.1"})
    with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as resp:
        data = resp.read()
    out.write_bytes(data)
    return out


def load_bybit_trades(path: Path) -> pd.DataFrame:
    """Load one Bybit day-CSV and return a normalized trade frame.

    Returns columns: timestamp (UTC datetime64), price, size (base), turnover
    (USDT), is_buyer_taker (bool). Sign convention matches the Binance writer:
    ``Buy`` side means the **taker** bought (an aggressive market buy), so
    ``is_buyer_taker = True``.
    """
    with gzip.open(path, "rt") as fh:
        raw = pd.read_csv(fh, low_memory=False)
    ts = pd.to_datetime(raw["timestamp"].astype(float), unit="s", utc=True)
    is_buyer = raw["side"].astype(str).str.lower().eq("buy")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "price": pd.to_numeric(raw["price"], errors="coerce"),
            "size": pd.to_numeric(raw["size"], errors="coerce"),
            "turnover": pd.to_numeric(raw["foreignNotional"], errors="coerce"),
            "is_buyer_taker": is_buyer,
        }
    )
    df = df.dropna(subset=["timestamp", "price", "size", "turnover"]).reset_index(drop=True)
    return df


def bin_trades_to_cvd_bars(
    trades: pd.DataFrame,
    symbol: str,
    bar_size: str,
    large_threshold: float | None,
    expected_bars: int,
) -> pd.DataFrame:
    """Bucket a normalized trade frame into fixed-size bars and emit the
    Binance-parity CVD schema.

    ``large_threshold`` is the turnover quantile cutoff for "large" trades; if
    None we compute it from the day's distribution (matches Binance writer).
    ``expected_bars`` is the number of bars one full day should produce at this
    resolution — used for ``coverage_ratio`` and the ``source_quality`` tag.
    """
    if trades.empty:
        return _empty_cvd_frame(symbol, bar_size)
    if large_threshold is None:
        large_threshold = float(
            np.quantile(trades["turnover"].to_numpy(), LARGE_TRADE_TURNOVER_QUANTILE)
        )

    trades = trades.assign(
        buy_volume=np.where(trades["is_buyer_taker"], trades["size"], 0.0),
        sell_volume=np.where(~trades["is_buyer_taker"], trades["size"], 0.0),
        buy_turnover=np.where(trades["is_buyer_taker"], trades["turnover"], 0.0),
        sell_turnover=np.where(~trades["is_buyer_taker"], trades["turnover"], 0.0),
        is_large=trades["turnover"] >= large_threshold,
    )
    trades["large_buy"] = trades["is_large"] & trades["is_buyer_taker"]
    trades["large_sell"] = trades["is_large"] & ~trades["is_buyer_taker"]
    trades["large_buy_turnover"] = np.where(trades["large_buy"], trades["turnover"], 0.0)
    trades["large_sell_turnover"] = np.where(trades["large_sell"], trades["turnover"], 0.0)

    floor = trades.set_index("timestamp").index.floor(bar_size)
    grouped = trades.assign(bar_open_time=floor).groupby("bar_open_time", sort=True)
    agg = grouped.agg(
        trade_count=("size", "size"),
        volume=("size", "sum"),
        turnover=("turnover", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        buy_turnover=("buy_turnover", "sum"),
        sell_turnover=("sell_turnover", "sum"),
        large_buy_count=("large_buy", "sum"),
        large_sell_count=("large_sell", "sum"),
        large_buy_turnover=("large_buy_turnover", "sum"),
        large_sell_turnover=("large_sell_turnover", "sum"),
    )
    agg = agg.reset_index()
    agg["symbol"] = symbol
    agg["bar_size"] = bar_size
    agg["taker_buy_ratio"] = agg["buy_volume"] / agg["volume"].replace(0, np.nan)
    agg["buy_sell_imbalance"] = (agg["buy_volume"] - agg["sell_volume"]) / agg["volume"].replace(
        0, np.nan
    )
    agg["cvd_delta_volume"] = agg["buy_volume"] - agg["sell_volume"]
    agg["cvd_delta_turnover"] = agg["buy_turnover"] - agg["sell_turnover"]
    agg["large_trade_threshold"] = large_threshold

    actual_bars = len(agg)
    coverage_ratio = float(actual_bars) / float(expected_bars) if expected_bars else 0.0
    agg["coverage_ratio"] = min(1.0, coverage_ratio)
    agg["source_quality"] = "complete" if coverage_ratio >= 0.98 else "partial"

    cols = [
        "symbol",
        "bar_open_time",
        "bar_size",
        "trade_count",
        "volume",
        "turnover",
        "buy_volume",
        "sell_volume",
        "buy_turnover",
        "sell_turnover",
        "taker_buy_ratio",
        "buy_sell_imbalance",
        "cvd_delta_volume",
        "cvd_delta_turnover",
        "large_trade_threshold",
        "large_buy_count",
        "large_sell_count",
        "large_buy_turnover",
        "large_sell_turnover",
        "coverage_ratio",
        "source_quality",
    ]
    return agg[cols]


def _empty_cvd_frame(symbol: str, bar_size: str) -> pd.DataFrame:
    cols = [
        "symbol",
        "bar_open_time",
        "bar_size",
        "trade_count",
        "volume",
        "turnover",
        "buy_volume",
        "sell_volume",
        "buy_turnover",
        "sell_turnover",
        "taker_buy_ratio",
        "buy_sell_imbalance",
        "cvd_delta_volume",
        "cvd_delta_turnover",
        "large_trade_threshold",
        "large_buy_count",
        "large_sell_count",
        "large_buy_turnover",
        "large_sell_turnover",
        "coverage_ratio",
        "source_quality",
    ]
    return pd.DataFrame(columns=cols)


def _expected_bar_count(bar_size: str) -> int:
    parsed = pd.Timedelta(bar_size)
    return int(pd.Timedelta(days=1) / parsed)


def backfill_symbol_day(
    cfg: BybitCvdConfig,
    symbol: str,
    day: date,
) -> dict[str, Path]:
    """Build and append continuous CVD for one (symbol, day) at all configured
    bar sizes. Idempotent — rewrites the affected month parquet with the union
    of existing rows and the new day, dropping duplicates by (symbol, bar_open_time)."""
    raw = download_bybit_day(cfg, symbol, day)
    trades = load_bybit_trades(raw)

    outputs: dict[str, Path] = {}
    month_key = day.strftime("%Y-%m")
    for bar_size in cfg.bar_sizes:
        expected = _expected_bar_count(bar_size)
        day_frame = bin_trades_to_cvd_bars(
            trades, symbol=symbol, bar_size=bar_size, large_threshold=None, expected_bars=expected
        )
        out_path = continuous_path(cfg, symbol, bar_size, month_key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            combined = pd.concat([existing, day_frame], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "bar_open_time"], keep="last"
            ).sort_values("bar_open_time").reset_index(drop=True)
            combined.to_parquet(out_path, index=False)
        else:
            day_frame.sort_values("bar_open_time").reset_index(drop=True).to_parquet(
                out_path, index=False
            )
        outputs[bar_size] = out_path
    return outputs


def backfill_symbol_range(
    cfg: BybitCvdConfig,
    symbol: str,
    start: date,
    end_inclusive: date,
    *,
    skip_existing_day: bool = True,
) -> list[dict[str, Path]]:
    """Sweep a date range, calling ``backfill_symbol_day`` per day."""
    results: list[dict[str, Path]] = []
    cur = start
    while cur <= end_inclusive:
        if skip_existing_day and raw_path(cfg, symbol, cur).exists():
            results.append({})
            cur += timedelta(days=1)
            continue
        try:
            results.append(backfill_symbol_day(cfg, symbol, cur))
        except urllib.error.HTTPError as exc:
            # 404 means Bybit hasn't published that day yet (weekend lag /
            # delisted symbol) — record an empty result and move on.
            results.append({"http_error": Path(str(exc))})
        cur += timedelta(days=1)
    return results
