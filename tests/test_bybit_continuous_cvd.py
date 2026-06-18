"""Unit tests for the Bybit continuous CVD pipeline."""
from __future__ import annotations

import gzip
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressure_graph.bybit_continuous_cvd import (
    BybitCvdConfig,
    _expected_bar_count,
    backfill_symbol_day,
    bin_trades_to_cvd_bars,
    load_bybit_trades,
)


def _synthetic_bybit_csv(path: Path) -> None:
    """Write a small synthetic Bybit-format gzip CSV with known trades."""
    rows = []
    base_ts = 1_700_000_000.0  # arbitrary epoch seconds
    for i in range(120):
        # alternating buy/sell, size 1 BTC at price 30000 + i
        side = "Buy" if i % 2 == 0 else "Sell"
        rows.append(
            {
                "timestamp": base_ts + i,
                "symbol": "BTCUSDT",
                "side": side,
                "size": 1.0,
                "price": 30_000.0 + i,
                "tickDirection": "ZeroPlusTick",
                "trdMatchID": f"id-{i}",
                "grossValue": 0,
                "homeNotional": 1.0,
                "foreignNotional": 30_000.0 + i,
                "RPI": 0,
            }
        )
    raw = pd.DataFrame(rows)
    with gzip.open(path, "wt") as fh:
        raw.to_csv(fh, index=False)


def test_load_bybit_trades_normalizes_side(tmp_path):
    csv = tmp_path / "sample.csv.gz"
    _synthetic_bybit_csv(csv)
    df = load_bybit_trades(csv)
    assert {"timestamp", "price", "size", "turnover", "is_buyer_taker"} <= set(df.columns)
    assert df["is_buyer_taker"].sum() == 60  # half the rows
    assert df["timestamp"].dt.tz is not None  # UTC tz-aware


def test_bin_trades_emits_binance_parity_schema(tmp_path):
    csv = tmp_path / "sample.csv.gz"
    _synthetic_bybit_csv(csv)
    trades = load_bybit_trades(csv)
    cvd = bin_trades_to_cvd_bars(
        trades, symbol="BTCUSDT", bar_size="1min", large_threshold=None, expected_bars=1440
    )
    expected_cols = {
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
    }
    assert set(cvd.columns) == expected_cols
    # synthetic data: alternating buy/sell, so taker_buy_ratio ≈ 0.5 per bar
    assert np.allclose(cvd["taker_buy_ratio"], 0.5)
    # cvd_delta_volume = buy - sell = 0 because perfectly alternating
    assert np.allclose(cvd["cvd_delta_volume"], 0.0)


def test_bin_trades_empty_returns_empty_frame_with_schema():
    empty = pd.DataFrame(
        columns=["timestamp", "price", "size", "turnover", "is_buyer_taker"]
    )
    cvd = bin_trades_to_cvd_bars(empty, "BTCUSDT", "1min", None, 1440)
    assert cvd.empty
    assert "cvd_delta_volume" in cvd.columns


def test_expected_bar_count_for_common_sizes():
    assert _expected_bar_count("1min") == 1440
    assert _expected_bar_count("5min") == 288
    assert _expected_bar_count("15min") == 96


def test_backfill_symbol_day_writes_and_is_idempotent(tmp_path, monkeypatch):
    """End-to-end test: pre-cache the raw CSV (so no network call), then run
    backfill_symbol_day twice and confirm parquets exist + are deduplicated."""
    cfg = BybitCvdConfig(history_root=tmp_path)
    sym, day = "BTCUSDT", date(2025, 10, 15)
    raw_csv = tmp_path / "raw" / "trades" / sym / f"{sym}{day.isoformat()}.csv.gz"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    _synthetic_bybit_csv(raw_csv)

    outputs1 = backfill_symbol_day(cfg, sym, day)
    outputs2 = backfill_symbol_day(cfg, sym, day)

    for bar_size, path in outputs1.items():
        assert path.exists()
        df1 = pd.read_parquet(path)
        df2 = pd.read_parquet(outputs2[bar_size])
        # Idempotent — same row count after the second run.
        assert len(df1) == len(df2)
        # No duplicate (symbol, bar_open_time) pairs.
        assert not df2.duplicated(subset=["symbol", "bar_open_time"]).any()
