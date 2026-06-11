from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from pressure_graph.clients.base import HttpClientConfig, RestClient
from pressure_graph.clients.bybit import to_ms


class BinanceClient(RestClient):
    exchange = "binance"

    def __init__(self, base_url: str = "https://fapi.binance.com") -> None:
        super().__init__(HttpClientConfig(base_url=base_url))

    def instruments(self) -> pd.DataFrame:
        payload = self.get_json("/fapi/v1/exchangeInfo")
        rows = [row for row in payload.get("symbols", []) if row.get("contractType") == "PERPETUAL"]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        if "onboardDate" in df.columns:
            df["launch_time"] = pd.to_datetime(pd.to_numeric(df["onboardDate"]), unit="ms", utc=True)
        else:
            df["launch_time"] = pd.NaT
        df["funding_interval_minutes"] = 480
        return df

    def tickers(self) -> pd.DataFrame:
        df = pd.DataFrame(self.get_json("/fapi/v1/ticker/24hr"))
        if df.empty:
            return df
        df = df[df["symbol"].str.endswith("USDT")].copy()
        df["exchange"] = self.exchange
        for col in ["quoteVolume", "volume", "lastPrice"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["turnover24h"] = df.get("quoteVolume")
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
        interval_ms = int(pd.Timedelta(interval).total_seconds() * 1000)
        cursor_start = start_ms
        while cursor_start < end_ms:
            batch = self.get_json(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor_start,
                    "endTime": end_ms,
                    "limit": 1500,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_start = int(batch[-1][0]) + interval_ms
            if next_start <= cursor_start:
                break
            cursor_start = next_start
        columns = [
            "bar_open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_raw",
            "turnover",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        df["symbol"] = symbol
        df["bar_open_time"] = pd.to_datetime(pd.to_numeric(df["bar_open_time"]), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["bar_close_time"] = df["bar_open_time"] + pd.Timedelta(interval)
        keep = ["exchange", "symbol", "bar_open_time", "bar_close_time", "open", "high", "low", "close", "volume", "turnover"]
        return df[keep].sort_values("bar_open_time").drop_duplicates(["symbol", "bar_open_time"])

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
            batch = self.get_json(
                "/fapi/v1/fundingRate",
                {
                    "symbol": symbol,
                    "startTime": cursor_start,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_start = max(int(row["fundingTime"]) for row in batch) + 1
            if next_start <= cursor_start:
                break
            cursor_start = next_start
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        df["symbol"] = symbol
        df["funding_time"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
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
        cursor_start = to_ms(pd.Timestamp(start).tz_convert("UTC"))
        end_ms = to_ms(pd.Timestamp(end).tz_convert("UTC"))
        # Binance documents this endpoint as latest-1-month only; in practice the
        # exact 30-day boundary can return HTTP 400, so keep the OI request inside it.
        max_oi_span_ms = int(pd.Timedelta(days=29).total_seconds() * 1000)
        cursor_start = max(cursor_start, end_ms - max_oi_span_ms)
        period = interval
        while cursor_start < end_ms:
            batch = self.get_json(
                "/futures/data/openInterestHist",
                {
                    "symbol": symbol,
                    "period": period,
                    "startTime": cursor_start,
                    "endTime": end_ms,
                    "limit": 500,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_start = max(int(row["timestamp"]) for row in batch) + 1
            if next_start <= cursor_start:
                break
            cursor_start = next_start
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["exchange"] = self.exchange
        df["symbol"] = symbol
        df["oi_time"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
        df["oi_base"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
        df["oi_value_usdt"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        return df[["exchange", "symbol", "oi_time", "oi_base", "oi_value_usdt"]].sort_values(
            "oi_time"
        )
