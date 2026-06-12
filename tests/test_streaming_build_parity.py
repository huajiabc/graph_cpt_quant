from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressure_graph.config import load_config
from pressure_graph.features.build import build_feature_table
from pressure_graph.features.streaming_build import (
    StreamingBuildConfig,
    build_v03_features_streaming,
)
from pressure_graph.labels.future import add_future_labels
from pressure_graph.universe.selection import apply_universe_flags

SYMBOLS = ["AAAUSDT", "BTCUSDT", "CCCUSDT"]
BARS = 4032  # 42 days of 15m bars: enough to exit the percentile warmup.
START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")


def _write_raw_layout(data_root: Path) -> None:
    rng = np.random.default_rng(11)
    times = pd.date_range(START, periods=BARS, freq="15min")
    for offset, symbol in enumerate(SYMBOLS):
        steps = rng.normal(0.0002, 0.004, BARS)
        close = 100.0 * (1.0 + offset / 10.0) * np.exp(np.cumsum(steps))
        high = close * (1.0 + np.abs(rng.normal(0, 0.002, BARS)))
        low = close * (1.0 - np.abs(rng.normal(0, 0.002, BARS)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = np.abs(rng.normal(1_000, 200, BARS))
        klines = pd.DataFrame(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "bar_open_time": times,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "turnover": volume * close,
            }
        )
        funding_times = pd.date_range(START, periods=BARS // 32, freq="8h")
        funding = pd.DataFrame(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "funding_time": funding_times,
                "funding_rate_settled": rng.normal(0.0001, 0.0002, len(funding_times)),
            }
        )
        oi = pd.DataFrame(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "oi_time": times,
                "oi_base": np.abs(rng.normal(5_000, 500, BARS)),
            }
        )
        for kind, frame in [("klines", klines), ("funding", funding), ("open_interest", oi)]:
            path = data_root / "raw" / "bybit" / kind / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
    instruments = pd.DataFrame(
        {
            "symbol": SYMBOLS,
            "exchange": "bybit",
            "launch_time": [START - pd.Timedelta(days=900)] * len(SYMBOLS),
            "funding_interval_minutes": [480.0] * len(SYMBOLS),
        }
    )
    tickers = pd.DataFrame(
        {
            "symbol": SYMBOLS,
            "exchange": "bybit",
            "turnover24h": [3e8, 9e9, 1e8],
        }
    )
    instruments.to_parquet(data_root / "raw" / "bybit" / "instruments.parquet", index=False)
    tickers.to_parquet(data_root / "raw" / "bybit" / "tickers.parquet", index=False)


def _test_config(tmp_path: Path):
    template = Path("configs/v0_3.yaml").read_text(encoding="utf-8")
    template = template.replace("data_root: data", f"data_root: {tmp_path.as_posix()}/data")
    template = template.replace(
        "report_root: reports/v0_3", f"report_root: {tmp_path.as_posix()}/reports/v0_3"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(template, encoding="utf-8")
    return load_config(config_path)


@pytest.mark.slow
def test_streaming_build_matches_monolithic(tmp_path):
    data_root = tmp_path / "data"
    _write_raw_layout(data_root)
    config = _test_config(tmp_path)

    instruments = pd.read_parquet(data_root / "raw" / "bybit" / "instruments.parquet")
    tickers = pd.read_parquet(data_root / "raw" / "bybit" / "tickers.parquet")
    klines = pd.concat(
        [
            pd.read_parquet(data_root / "raw" / "bybit" / "klines" / f"{symbol}.parquet")
            for symbol in SYMBOLS
        ],
        ignore_index=True,
    )
    funding = pd.concat(
        [
            pd.read_parquet(data_root / "raw" / "bybit" / "funding" / f"{symbol}.parquet")
            for symbol in SYMBOLS
        ],
        ignore_index=True,
    )
    oi = pd.concat(
        [
            pd.read_parquet(data_root / "raw" / "bybit" / "open_interest" / f"{symbol}.parquet")
            for symbol in SYMBOLS
        ],
        ignore_index=True,
    )
    flagged = apply_universe_flags(klines, instruments, tickers, config)
    monolithic = add_future_labels(build_feature_table(flagged, funding, oi, instruments, config))
    monolithic = monolithic.sort_values(["exchange", "symbol", "bar_open_time"]).reset_index(
        drop=True
    )

    output = build_v03_features_streaming(
        config, StreamingBuildConfig(batch_size=2, workers=1)
    )
    streaming = pd.read_parquet(output)
    streaming = streaming.sort_values(["exchange", "symbol", "bar_open_time"]).reset_index(
        drop=True
    )

    assert set(monolithic.columns) == set(streaming.columns)
    streaming = streaming[monolithic.columns]
    # The BTC volatility percentile carry-over artifact must match exactly,
    # including the contaminated head of every non-first symbol segment.
    pd.testing.assert_frame_equal(streaming, monolithic, check_dtype=False)

    # Carry-over artifact: a non-first segment turns finite as soon as the BTC
    # volatility input itself is finite (history window arrives pre-filled),
    # while the first segment must wait out the full min_periods warmup.
    btc_pct = monolithic[monolithic["symbol"].eq("BTCUSDT")][
        "btc_volatility_percentile"
    ].reset_index(drop=True)
    first_pct = monolithic[monolithic["symbol"].eq("AAAUSDT")][
        "btc_volatility_percentile"
    ].reset_index(drop=True)
    assert int(btc_pct.notna().idxmax()) < 100, "carried history should skip the warmup"
    assert int(first_pct.notna().idxmax()) >= 1344, "first segment starts cold"
