from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from pressure_graph.clients.base import HttpClientConfig, RestClient


class DeribitClient(RestClient):
    """Minimal public Deribit market-data client used by volatility research."""

    exchange = "deribit"

    def __init__(self, base_url: str = "https://www.deribit.com") -> None:
        super().__init__(HttpClientConfig(base_url=base_url, timeout=60.0))

    @staticmethod
    def _ms(value: datetime | pd.Timestamp) -> int:
        return int(pd.Timestamp(value).tz_convert("UTC").timestamp() * 1000)

    @staticmethod
    def _result(payload: dict[str, Any]) -> dict[str, Any]:
        error = payload.get("error")
        if error:
            raise RuntimeError(f"Deribit API error: {error}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Deribit API response has no object result")
        return result

    def volatility_index_data(
        self,
        currency: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        resolution_seconds: int = 3600,
    ) -> pd.DataFrame:
        payload = self.get_json(
            "/api/v2/public/get_volatility_index_data",
            {
                "currency": currency.upper(),
                "start_timestamp": self._ms(start),
                "end_timestamp": self._ms(end),
                "resolution": str(int(resolution_seconds)),
            },
        )
        result = self._result(payload)
        rows = result.get("data", [])
        frame = pd.DataFrame(rows, columns=["dvol_time", "open", "high", "low", "close"])
        if frame.empty:
            return frame
        frame["dvol_time"] = pd.to_datetime(
            pd.to_numeric(frame["dvol_time"], errors="coerce"), unit="ms", utc=True
        )
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["exchange"] = self.exchange
        frame["currency"] = currency.upper()
        return (
            frame.dropna(subset=["dvol_time", "close"])
            .drop_duplicates("dvol_time", keep="last")
            .sort_values("dvol_time")
            .reset_index(drop=True)
        )

    def chart_data(
        self,
        instrument_name: str,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        resolution_minutes: int = 60,
    ) -> pd.DataFrame:
        payload = self.get_json(
            "/api/v2/public/get_tradingview_chart_data",
            {
                "instrument_name": instrument_name,
                "start_timestamp": self._ms(start),
                "end_timestamp": self._ms(end),
                "resolution": str(int(resolution_minutes)),
            },
        )
        result = self._result(payload)
        if result.get("status") != "ok":
            return pd.DataFrame()
        ticks = result.get("ticks", [])
        frame = pd.DataFrame(
            {
                "bar_open_time": ticks,
                "open": result.get("open", []),
                "high": result.get("high", []),
                "low": result.get("low", []),
                "close": result.get("close", []),
                "volume": result.get("volume", []),
                "cost": result.get("cost", []),
            }
        )
        if frame.empty:
            return frame
        frame["bar_open_time"] = pd.to_datetime(
            pd.to_numeric(frame["bar_open_time"], errors="coerce"), unit="ms", utc=True
        )
        for column in ("open", "high", "low", "close", "volume", "cost"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["exchange"] = self.exchange
        frame["instrument_name"] = instrument_name
        return (
            frame.dropna(subset=["bar_open_time", "close"])
            .drop_duplicates("bar_open_time", keep="last")
            .sort_values("bar_open_time")
            .reset_index(drop=True)
        )


__all__ = ["DeribitClient"]
