from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v40_cross_exchange_lead_lag import (
    V40Config,
    write_v40_cross_exchange_lead_lag,
)


def _write_klines(root: Path, exchange: str, symbol: str, closes: list[float], volumes: list[float]) -> None:
    times = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    prev = [closes[0], *closes[:-1]]
    rows = []
    for idx, close in enumerate(closes):
        open_price = prev[idx]
        high = max(open_price, close) * 1.002
        low = min(open_price, close) * 0.998
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "bar_open_time": times[idx],
                "bar_close_time": times[idx] + pd.Timedelta(minutes=15),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volumes[idx],
                "turnover": volumes[idx] * close,
            }
        )
    out = root / "raw" / exchange / "klines"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / f"{symbol}.parquet", index=False)


def test_v40_cross_exchange_reports_lead_lag_motif(tmp_path: Path) -> None:
    source_aaa = [100, 100, 100, 100, 100, 102, 102.2, 102.1, 102.0, 102.4, 102.5, 102.6]
    target_aaa = [100, 100, 100, 100, 100, 99, 100.5, 101.0, 102.0, 103.0, 104.0, 104.2]
    source_bbb = [50, 50, 50.1, 50.0, 50.1, 50.0, 50.2, 50.1, 50.0, 50.2, 50.1, 50.0]
    target_bbb = [50, 50, 49.9, 50.0, 50.1, 50.0, 50.1, 50.0, 50.1, 50.0, 50.1, 50.0]
    base_vol = [900.0, 1000.0, 1100.0, 950.0, 1050.0, 1000.0, 980.0, 1020.0, 990.0, 1010.0, 970.0, 1030.0]
    source_vol = base_vol.copy()
    source_vol[5] = 10000.0
    _write_klines(tmp_path, "binance", "AAAUSDT", source_aaa, source_vol)
    _write_klines(tmp_path, "bybit", "AAAUSDT", target_aaa, base_vol)
    _write_klines(tmp_path, "binance", "BBBUSDT", source_bbb, base_vol)
    _write_klines(tmp_path, "bybit", "BBBUSDT", target_bbb, base_vol)

    outputs = write_v40_cross_exchange_lead_lag(
        V40Config(
            report_root=tmp_path / "reports",
            data_root=tmp_path,
            top_n=10,
            max_pair_symbols=2,
            impulse_volume_z=1.0,
            impulse_ret_threshold=0.005,
        )
    )

    assert outputs["cross_exchange_edge_atlas"].exists()
    assert outputs["lead_lag_motif_summary"].exists()
    assert outputs["random_shuffled_exchange_control"].exists()
    motifs = pd.read_csv(outputs["lead_lag_motif_summary"])
    lx1 = motifs.loc[motifs["label"].eq("LX1_source_impulse_target_lag_reclaim")].iloc[0]
    assert int(lx1["events"]) >= 1
    assert float(lx1["net20"]) > 0
    coverage = pd.read_csv(outputs["coverage_summary"])
    assert int(coverage["common_symbols_available"].iloc[0]) == 2
    controls = pd.read_csv(outputs["random_shuffled_exchange_control"])
    assert "random_cyclic_source_symbol" in set(controls["control_type"])
