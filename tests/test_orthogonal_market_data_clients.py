from __future__ import annotations

import pandas as pd

from pressure_graph.clients.bybit import BybitClient
from pressure_graph.clients.deribit import DeribitClient


def test_bybit_account_ratio_paginates_and_normalizes(monkeypatch) -> None:
    client = BybitClient(base_url="https://example.invalid")
    calls = []

    def fake_get_json(path, params):
        calls.append((path, params.copy()))
        cursor = params.get("cursor")
        if cursor is None:
            return {
                "result": {
                    "list": [
                        {"timestamp": "3600000", "buyRatio": "0.6", "sellRatio": "0.4"}
                    ],
                    "nextPageCursor": "page-2",
                }
            }
        return {
            "result": {
                "list": [
                    {"timestamp": "0", "buyRatio": "0.5", "sellRatio": "0.5"}
                ],
                "nextPageCursor": "",
            }
        }

    monkeypatch.setattr(client, "get_json", fake_get_json)
    try:
        frame = client.account_ratio(
            "BTCUSDT",
            pd.Timestamp("1970-01-01", tz="UTC"),
            pd.Timestamp("1970-01-02", tz="UTC"),
        )
    finally:
        client.close()
    assert len(calls) == 2
    assert calls[1][1]["cursor"] == "page-2"
    assert frame["account_ratio_time"].is_monotonic_increasing
    assert frame["long_account_ratio"].tolist() == [0.5, 0.6]


def test_deribit_dvol_normalizes_rows(monkeypatch) -> None:
    client = DeribitClient(base_url="https://example.invalid")
    monkeypatch.setattr(
        client,
        "get_json",
        lambda path, params: {
            "result": {"data": [[0, 40, 42, 39, 41], [3600000, 41, 43, 40, 42]]}
        },
    )
    try:
        frame = client.volatility_index_data(
            "btc",
            pd.Timestamp("1970-01-01", tz="UTC"),
            pd.Timestamp("1970-01-02", tz="UTC"),
        )
    finally:
        client.close()
    assert frame["currency"].eq("BTC").all()
    assert frame["close"].tolist() == [41, 42]
    assert frame["dvol_time"].is_monotonic_increasing


def test_deribit_chart_handles_no_data(monkeypatch) -> None:
    client = DeribitClient(base_url="https://example.invalid")
    monkeypatch.setattr(client, "get_json", lambda path, params: {"result": {"status": "no_data"}})
    try:
        frame = client.chart_data(
            "BTCDVOL_USDC-31MAY23",
            pd.Timestamp("2023-05-01", tz="UTC"),
            pd.Timestamp("2023-05-31", tz="UTC"),
        )
    finally:
        client.close()
    assert frame.empty
