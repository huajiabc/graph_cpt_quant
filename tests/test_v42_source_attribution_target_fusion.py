from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v42_source_attribution_target_fusion import (
    V42Config,
    write_v42_source_attribution_target_fusion,
)


def _write_1m(root: Path, exchange: str, symbol: str, closes: list[float], volumes: list[float]) -> None:
    opens = [closes[0], *closes[:-1]]
    times = pd.date_range("2026-01-01", periods=len(closes), freq="1min", tz="UTC")
    rows = []
    for idx, close in enumerate(closes):
        open_price = opens[idx]
        buy_ratio = 0.72 if exchange == "binance" and idx in {40, 41} else 0.45
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "bar_open_time": times[idx],
                "bar_close_time": times[idx] + pd.Timedelta(minutes=1),
                "open": open_price,
                "high": max(open_price, close) * 1.001,
                "low": min(open_price, close) * 0.999,
                "close": close,
                "volume": volumes[idx],
                "turnover": volumes[idx] * close,
                "trades": 100 + idx,
                "taker_buy_base": volumes[idx] * buy_ratio,
                "taker_buy_quote": volumes[idx] * close * buy_ratio,
            }
        )
    out = root / "raw" / exchange / "klines_1m_v4"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / f"{symbol}.parquet", index=False)


def _series(base: float, jump_idx: int | None = None) -> list[float]:
    closes = []
    price = base
    for idx in range(130):
        if idx == jump_idx:
            price *= 1.004
        elif idx == 39:
            price *= 0.998
        elif idx == 40:
            price *= 1.003
        elif idx > 40:
            price *= 1.0002
        closes.append(price)
    return closes


def _volumes(spike_idx: int | None = None) -> list[float]:
    out = [1000.0 + (idx % 7) * 15.0 for idx in range(130)]
    if spike_idx is not None:
        out[spike_idx] = 6500.0
    return out


def _write_trade_cache(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "exchange": "bybit",
            "symbol": "AAAUSDT",
            "candidate": "CIC1_beta_extreme",
            "signal_time": "2026-01-01 00:40:00+00:00",
            "entry_time": "2026-01-01 00:41:00+00:00",
            "exit_time": "2026-01-01 02:00:00+00:00",
            "gross_return": 0.03,
            "net_return": 0.028,
            "cost_single_side_bps": 10.0,
            "base_signal_id": "AAA|1",
            "volume_impulse_density": 0.4,
        },
        {
            "exchange": "bybit",
            "symbol": "BBBUSDT",
            "candidate": "CIC2_beta_broad",
            "signal_time": "2026-01-01 00:42:00+00:00",
            "entry_time": "2026-01-01 00:43:00+00:00",
            "exit_time": "2026-01-01 02:00:00+00:00",
            "gross_return": -0.01,
            "net_return": -0.012,
            "cost_single_side_bps": 10.0,
            "base_signal_id": "BBB|1",
            "volume_impulse_density": 0.2,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_v42_writes_attribution_and_fusion_tables(tmp_path: Path) -> None:
    _write_1m(tmp_path, "binance", "AAAUSDT", _series(100.0, 40), _volumes(40))
    _write_1m(tmp_path, "bybit", "AAAUSDT", _series(100.0), _volumes(None))
    _write_1m(tmp_path, "binance", "BBBUSDT", _series(50.0), _volumes(None))
    _write_1m(tmp_path, "bybit", "BBBUSDT", _series(50.0), _volumes(None))
    trade_cache = tmp_path / "reports" / "trades.csv"
    _write_trade_cache(trade_cache)

    outputs = write_v42_source_attribution_target_fusion(
        V42Config(report_root=tmp_path / "reports" / "v42", data_root=tmp_path, trade_cache_path=trade_cache, top_n=2)
    )

    attribution = pd.read_csv(outputs["source_target_attribution_1m"])
    assert "A2_binance_source_plus_bybit_reclaim" in set(attribution["candidate"])
    a2 = attribution[attribution["candidate"].eq("A2_binance_source_plus_bybit_reclaim")].iloc[0]
    assert int(a2["events"]) >= 1

    fusion = pd.read_csv(outputs["cross_exchange_fusion_with_cic"])
    assert "P2_source_context_covered" in set(fusion["bucket"])
    assert int(fusion.loc[fusion["bucket"].eq("P2_all_trades"), "trades"].iloc[0]) == 2
