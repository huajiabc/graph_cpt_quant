from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.config.v05 import load_v05_config
from pressure_graph.paper_live import build_v05_paper_ledger
from pressure_graph.reports.v03 import C2_SIGNAL_COL


def _rows(symbol: str, state: str, rank90: int, start: str = "2026-05-01 00:00:00Z") -> list[dict]:
    times = pd.date_range(start, periods=5, freq="15min", tz="UTC")
    lows = [100.0, 99.40, 100.0, 100.0, 100.0]
    highs = [100.0, 101.0, 105.0, 101.0, 101.0]
    rows = []
    for idx, ts in enumerate(times):
        rows.append(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "bar_open_time": ts,
                "bar_close_time": ts + pd.Timedelta(minutes=15),
                "feature_time": ts + pd.Timedelta(minutes=15),
                "open": 100.0,
                "high": highs[idx],
                "low": lows[idx],
                "close": 100.0,
                "btc_market_state": state,
                "btc_ret_1h": 0.01,
                "btc_ret_4h": 0.02,
                "btc_volatility_4h": 0.01,
                "turnover_rank_30d": 10,
                "turnover_rank_90d": rank90,
                "core_liquidity": rank90 <= 50,
                "transient_hot": rank90 > 50,
                "trailing_30d_turnover": 1_000_000.0,
                "oi_value_delta_1h_percentile": 80,
                "oi_value_delta_4h_percentile": 80,
                "funding_percentile": 50,
                "funding_z": 0.1,
                "ret_1h": 0.01,
                "ret_4h": 0.02,
                "volume_z_1h": 1.5,
                "volume_z_4h": 1.2,
                C2_SIGNAL_COL: idx == 0,
            }
        )
    return rows


def test_v05_gates_and_pullback_lifecycle() -> None:
    cfg = load_v05_config()
    prepared = pd.DataFrame(
        [
            *_rows("AAAUSDT", "BTC_up", 20),
            *_rows("BBBUSDT", "BTC_chop", 20),
            *_rows("CCCUSDT", "BTC_up", 80),
        ]
    )

    signals, trades, shadow = build_v05_paper_ledger(prepared, cfg)

    assert len(signals) == 3
    assert len(trades) == 1
    assert set(signals["status"]) == {"exited", "skipped"}
    assert "btc_not_up:BTC_chop" in set(signals["skip_reason"])
    assert "transient_hot" in set(signals["skip_reason"])

    trade = trades.iloc[0]
    assert trade["symbol"] == "AAAUSDT"
    assert trade["exit_reason"] == "tp"
    assert trade["gross_return"] == pytest.approx(0.05)
    assert trade["net_return_10bp"] == pytest.approx(0.048)
    assert pd.notna(trade["fill_time_conservative_5bp"])
    assert not shadow.empty


def test_v05_signal_start_time_accepts_utc_timestamp() -> None:
    cfg = load_v05_config()
    prepared = pd.DataFrame(_rows("AAAUSDT", "BTC_up", 20))

    signals, trades, _ = build_v05_paper_ledger(
        prepared,
        cfg,
        signal_start_time=pd.Timestamp("2026-05-01 00:15:00Z"),
    )

    assert len(signals) == 1
    assert len(trades) == 1
