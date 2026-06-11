from __future__ import annotations

import gzip
import io
from pathlib import Path

import httpx
import pandas as pd

from pressure_graph.io import ensure_dir


PUBLIC_BYBIT_BASE = "https://public.bybit.com"


def public_trading_url(symbol: str, date: pd.Timestamp) -> str:
    day = pd.Timestamp(date).strftime("%Y-%m-%d")
    return f"{PUBLIC_BYBIT_BASE}/trading/{symbol}/{symbol}{day}.csv.gz"


def download_public_trading_day(
    symbol: str,
    date: pd.Timestamp,
    cache_root: Path,
    timeout: float = 120.0,
) -> Path | None:
    cache_dir = ensure_dir(cache_root / symbol)
    day = pd.Timestamp(date).strftime("%Y-%m-%d")
    out = cache_dir / f"{symbol}{day}.csv.gz"
    if out.exists() and out.stat().st_size > 0:
        return out
    response = httpx.get(public_trading_url(symbol, pd.Timestamp(date)), timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(response.content)
    tmp.replace(out)
    return out


def public_trades_to_1m_ohlcv(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as fh:
        payload = fh.read()
    trades = pd.read_csv(io.BytesIO(payload))
    trades = normalize_public_trades(trades)
    if trades.empty:
        return pd.DataFrame()
    trades["bar_open_time"] = trades["timestamp"].dt.floor("1min")
    symbol = str(trades["symbol"].iloc[0])
    grouped = trades.groupby("bar_open_time", sort=True)
    out = grouped["price"].ohlc().reset_index()
    out["volume"] = grouped["size"].sum().to_numpy()
    out["turnover"] = grouped["turnover"].sum().to_numpy()
    out["exchange"] = "bybit"
    out["symbol"] = symbol
    out["bar_close_time"] = out["bar_open_time"] + pd.Timedelta(minutes=1)
    return out[
        [
            "exchange",
            "symbol",
            "bar_open_time",
            "bar_close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ]
    ]


def normalize_public_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades["timestamp"] = pd.to_datetime(pd.to_numeric(trades["timestamp"], errors="coerce"), unit="s", utc=True)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["size"] = pd.to_numeric(trades["size"], errors="coerce")
    if "foreignNotional" in trades.columns:
        trades["turnover"] = pd.to_numeric(trades["foreignNotional"], errors="coerce")
    else:
        trades["turnover"] = trades["price"] * trades["size"]
    trades = trades.dropna(subset=["timestamp", "price"])
    if trades.empty:
        return pd.DataFrame()
    trades["exchange"] = "bybit"
    return trades[
        [
            "exchange",
            "symbol",
            "timestamp",
            "price",
            "size",
            "turnover",
            "side",
        ]
    ]


def load_public_trade_file(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as fh:
        payload = fh.read()
    return normalize_public_trades(pd.read_csv(io.BytesIO(payload)))
