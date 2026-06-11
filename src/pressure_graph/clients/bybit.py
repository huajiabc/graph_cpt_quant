from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from pressure_graph.clients.base import HttpClientConfig, RestClient


INTERVAL_TO_BYBIT = {
    "1m": "1",
    "15m": "15",
    "1h": "60",
    "4h": "240",
}

INTERVAL_TO_OI = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
}


def to_ms(ts: datetime | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).tz_convert("UTC").timestamp() * 1000)


class BybitClient(RestClient):
    exchange = "bybit"

    def __init__(self, base_url: str = "https://api.bybit.com", category: str = "linear") -> None:
        super().__init__(HttpClientConfig(base_url=base_url))
        self.category = category

    def instruments(self, settle_coin: str = "USDT") -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "category": self.category,
                "settleCoin": settle_coin,
                "status": "Trading",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self.get_json("/v5/market/instruments-info", params)
            result = payload["result"]
            rows.extend(result.get("list", []))
            cursor = result.get("nextPageCursor") or None
            if not cursor:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        df["launch_time"] = pd.to_datetime(pd.to_numeric(df["launchTime"]), unit="ms", utc=True)
        df["funding_interval_minutes"] = pd.to_numeric(
            df.get("fundingInterval", 480), errors="coerce"
        ).fillna(480)
        return df

    def tickers(self, settle_coin: str = "USDT") -> pd.DataFrame:
        payload = self.get_json("/v5/market/tickers", {"category": self.category})
        rows = payload["result"].get("list", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if "symbol" in df.columns:
            df = df[df["symbol"].str.endswith(settle_coin)].copy()
        df["exchange"] = self.exchange
        for col in ["turnover24h", "volume24h", "openInterest", "fundingRate", "lastPrice"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def klines(
        self,
        symbol: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        interval: str = "15m",
    ) -> pd.DataFrame:
        rows: list[list[Any]] = []
        start_ms = to_ms(pd.Timestamp(start).tz_convert("UTC"))
        end_ms = to_ms(pd.Timestamp(end).tz_convert("UTC"))
        cursor_start = start_ms
        interval_ms = int(pd.Timedelta(interval).total_seconds() * 1000)
        max_rows = 1000
        while cursor_start < end_ms:
            chunk_end = min(cursor_start + interval_ms * (max_rows - 1), end_ms)
            payload = self.get_json(
                "/v5/market/kline",
                {
                    "category": self.category,
                    "symbol": symbol,
                    "interval": INTERVAL_TO_BYBIT[interval],
                    "start": cursor_start,
                    "end": chunk_end,
                    "limit": max_rows,
                },
            )
            batch = payload["result"].get("list", [])
            if not batch:
                cursor_start = chunk_end + interval_ms
                continue
            rows.extend(batch)
            starts = [int(row[0]) for row in batch]
            next_start = max(starts) + interval_ms
            if next_start <= cursor_start:
                next_start = chunk_end + interval_ms
            cursor_start = next_start
        columns = ["bar_open_time", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        df["symbol"] = symbol
        df["bar_open_time"] = pd.to_datetime(pd.to_numeric(df["bar_open_time"]), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["bar_close_time"] = df["bar_open_time"] + pd.Timedelta(interval)
        return df.sort_values("bar_open_time").drop_duplicates(["symbol", "bar_open_time"])

    def funding_history(
        self,
        symbol: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cursor_start = to_ms(pd.Timestamp(start).tz_convert("UTC"))
        end_ms = to_ms(pd.Timestamp(end).tz_convert("UTC"))
        while cursor_start < end_ms:
            # This endpoint has no cursor and caps at 200 rows. A 7-day chunk stays
            # below that cap even for 1h funding-interval symbols.
            chunk_end = min(cursor_start + int(pd.Timedelta(days=7).total_seconds() * 1000), end_ms)
            payload = self.get_json(
                "/v5/market/funding/history",
                {
                    "category": self.category,
                    "symbol": symbol,
                    "startTime": cursor_start,
                    "endTime": chunk_end,
                    "limit": 200,
                },
            )
            batch = payload["result"].get("list", [])
            if not batch:
                cursor_start = chunk_end + 1
                continue
            rows.extend(batch)
            cursor_start = chunk_end + 1
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        df["symbol"] = symbol
        df["funding_time"] = pd.to_datetime(
            pd.to_numeric(df["fundingRateTimestamp"]), unit="ms", utc=True
        )
        df["funding_rate_settled"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df[["exchange", "symbol", "funding_time", "funding_rate_settled"]].sort_values(
            "funding_time"
        )

    def open_interest(
        self,
        symbol: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        interval: str = "15m",
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "category": self.category,
                "symbol": symbol,
                "intervalTime": INTERVAL_TO_OI[interval],
                "startTime": to_ms(pd.Timestamp(start).tz_convert("UTC")),
                "endTime": to_ms(pd.Timestamp(end).tz_convert("UTC")),
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self.get_json("/v5/market/open-interest", params)
            result = payload["result"]
            batch = result.get("list", [])
            rows.extend(batch)
            cursor = result.get("nextPageCursor") or None
            if not cursor:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        df["symbol"] = symbol
        df["oi_time"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        df["oi_base"] = pd.to_numeric(df["openInterest"], errors="coerce")
        return df[["exchange", "symbol", "oi_time", "oi_base"]].sort_values("oi_time")

    def recent_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        payload = self.get_json(
            "/v5/market/recent-trade",
            {
                "category": self.category,
                "symbol": symbol,
                "limit": int(limit),
            },
        )
        rows = payload["result"].get("list", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        time_col = "execTime" if "execTime" in df.columns else "time"
        price_col = "price" if "price" in df.columns else "p"
        size_col = "size" if "size" in df.columns else "v"
        side_col = "side" if "side" in df.columns else "S"
        df["exchange"] = self.exchange
        df["symbol"] = symbol
        df["timestamp"] = pd.to_datetime(
            pd.to_numeric(df[time_col], errors="coerce"), unit="ms", utc=True
        )
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")
        df["size"] = pd.to_numeric(df[size_col], errors="coerce")
        df["side"] = df[side_col].astype(str)
        df["turnover"] = df["price"] * df["size"]
        if "execId" not in df.columns:
            df["execId"] = (
                df["timestamp"].astype(str)
                + ":"
                + df["price"].astype(str)
                + ":"
                + df["size"].astype(str)
                + ":"
                + df["side"].astype(str)
            )
        return df[
            [
                "exchange",
                "symbol",
                "timestamp",
                "execId",
                "price",
                "size",
                "turnover",
                "side",
            ]
        ].dropna(subset=["timestamp", "price"]).sort_values("timestamp")

    def orderbook(self, symbol: str, limit: int = 200) -> dict[str, Any]:
        payload = self.get_json(
            "/v5/market/orderbook",
            {
                "category": self.category,
                "symbol": symbol,
                "limit": int(limit),
            },
        )
        return payload.get("result", {})


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))
