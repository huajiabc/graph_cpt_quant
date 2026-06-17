"""Tests for binance_continuous_cvd — pure-function feature aggregation."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.binance_continuous_cvd import (
    ContinuousCvdConfig,
    _append_shard,
    _shard_path,
    compute_continuous_features,
)


def _synth_trades(
    *,
    n: int,
    base: pd.Timestamp,
    side_pattern: list[str] | None = None,
    price: float = 100.0,
) -> pd.DataFrame:
    """Build a synthetic normalized trade frame.

    Default pattern: 60/40 buy/sell across n trades evenly spaced over 60 seconds.
    """
    sides = side_pattern or (["buy"] * (n * 6 // 10) + ["sell"] * (n - n * 6 // 10))
    while len(sides) < n:
        sides.append("buy")
    sides = sides[:n]
    timestamps = pd.date_range(base, periods=n, freq="1s", tz="UTC")
    sizes = np.full(n, 10.0)
    return pd.DataFrame(
        {
            "exchange": ["binance"] * n,
            "symbol": ["BTCUSDT"] * n,
            "timestamp": timestamps,
            "execId": [str(i) for i in range(n)],
            "price": [price] * n,
            "size": sizes,
            "turnover": [price * 10.0] * n,
            "side": sides,
        }
    )


class TestComputeContinuousFeatures:
    def test_empty_trades_returns_empty_schema(self) -> None:
        out = compute_continuous_features(
            pd.DataFrame(), symbol="BTCUSDT", day=date(2026, 1, 1), bar_size="1min"
        )
        assert out.empty
        assert "symbol" in out.columns
        assert "bar_open_time" in out.columns
        assert "taker_buy_ratio" in out.columns

    def test_one_minute_bin_balanced(self) -> None:
        # 60 trades over 60 seconds → 1 bar at 1min granularity
        trades = _synth_trades(n=60, base=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"))
        out = compute_continuous_features(trades, symbol="BTCUSDT", day=date(2026, 1, 1), bar_size="1min")
        assert len(out) == 1
        row = out.iloc[0]
        assert row["trade_count"] == 60
        assert row["bar_size"] == "1min"
        assert row["source_quality"] == "complete"
        # 60% buy → taker_buy_ratio ≈ 0.6
        assert abs(row["taker_buy_ratio"] - 0.6) < 0.02
        # imbalance: (buy_to - sell_to) / total ≈ 0.6 - 0.4 = 0.2
        assert abs(row["buy_sell_imbalance"] - 0.2) < 0.02
        assert row["volume"] > 0
        assert row["large_trade_threshold"] >= 0
        assert row["coverage_ratio"] > 0

    def test_imbalance_negative_when_sell_dominates(self) -> None:
        sides = ["sell"] * 80 + ["buy"] * 20
        trades = _synth_trades(
            n=100, base=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), side_pattern=sides
        )
        # Spread trades across 100s → first 60s = 1 bar, next 40s spills to next bar
        out = compute_continuous_features(trades, symbol="BTCUSDT", day=date(2026, 1, 1), bar_size="1min")
        # buy_sell_imbalance for the first bar should be negative.
        first = out.iloc[0]
        assert first["buy_sell_imbalance"] < 0
        assert first["taker_buy_ratio"] < 0.5

    def test_five_minute_bin_aggregates_one_minute_trades(self) -> None:
        trades = _synth_trades(n=300, base=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"))
        out = compute_continuous_features(trades, symbol="BTCUSDT", day=date(2026, 1, 1), bar_size="5min")
        assert len(out) == 1
        assert out.iloc[0]["trade_count"] == 300

    def test_drops_trades_outside_target_day(self) -> None:
        early = _synth_trades(n=30, base=pd.Timestamp("2025-12-31 23:30:00", tz="UTC"))
        on_day = _synth_trades(n=30, base=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"))
        trades = pd.concat([early, on_day], ignore_index=True)
        out = compute_continuous_features(trades, symbol="BTCUSDT", day=date(2026, 1, 1), bar_size="1min")
        # Only on-day trades retained.
        assert int(out["trade_count"].sum()) == 30


class TestShardLayout:
    def test_shard_path_includes_year_month(self, tmp_path: Path) -> None:
        cfg = ContinuousCvdConfig(continuous_root=tmp_path / "continuous")
        path = _shard_path(cfg.continuous_root, "BTCUSDT", date(2026, 3, 15), "1min")
        assert path.parts[-1] == "2026-03.parquet"
        assert path.parts[-2] == "1min"
        assert path.parts[-3] == "BTCUSDT"

    def test_append_shard_dedupes_by_bar_open_time(self, tmp_path: Path) -> None:
        trades = _synth_trades(n=60, base=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"))
        features = compute_continuous_features(
            trades, symbol="BTCUSDT", day=date(2026, 1, 1), bar_size="1min"
        )
        shard_path = tmp_path / "BTCUSDT" / "1min" / "2026-01.parquet"
        _append_shard(features, shard_path)
        _append_shard(features, shard_path)  # second write → same bars
        loaded = pd.read_parquet(shard_path)
        # Both writes had the same bar — dedup should keep one row.
        assert len(loaded) == 1


class TestBackfillDriver:
    def test_missing_zip_records_missing_outcome(self, tmp_path: Path) -> None:
        from pressure_graph.binance_continuous_cvd import backfill_symbol_day

        cfg = ContinuousCvdConfig(
            history_root=tmp_path / "history",
            continuous_root=tmp_path / "continuous",
            download_if_missing=False,
        )
        outcome = backfill_symbol_day("BTCUSDT", date(2026, 1, 1), cfg)
        assert outcome.source_quality == "missing"
        assert outcome.trade_count == 0
