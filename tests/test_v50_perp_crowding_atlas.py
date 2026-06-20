from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v50_perp_crowding_atlas import V50Config, write_v50_perp_crowding_atlas


def _feature_rows(symbol: str, base: float, high_crowding: bool) -> list[dict[str, object]]:
    times = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
    rows: list[dict[str, object]] = []
    for idx, ts in enumerate(times):
        close = base * (1.0 + 0.0005 * idx)
        future = 0.018 if high_crowding else 0.012
        if symbol == "BTCUSDT":
            future = 0.006
        rows.append(
            {
                "symbol": symbol,
                "feature_time": ts,
                "bar_close_time": ts,
                "close": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "volume": 1000.0 + idx,
                "turnover": close * (1000.0 + idx),
                "warmup_complete": True,
                "universe_dynamic_monthly_top30": True,
                "funding_rate_settled": 0.0002 if high_crowding else 0.00001,
                "funding_z": 2.0 if high_crowding else 0.1,
                "funding_percentile": 0.92 if high_crowding else 0.45,
                "oi_base": 10000.0,
                "oi_value_usdt": 1_000_000.0,
                "oi_delta_1h": 0.02 if high_crowding else 0.005,
                "oi_delta_4h": 0.08 if high_crowding else 0.02,
                "oi_value_delta_1h": 10_000.0,
                "oi_value_delta_4h": 40_000.0,
                "oi_value_delta_z_1h": 1.0,
                "oi_value_delta_z_4h": 2.0 if high_crowding else 0.5,
                "oi_value_delta_1h_percentile": 0.88 if high_crowding else 0.55,
                "oi_value_delta_4h_percentile": 0.91 if high_crowding else 0.62,
                "oi_delta_1h_percentile": 0.86 if high_crowding else 0.5,
                "oi_delta_4h_percentile": 0.9 if high_crowding else 0.58,
                "ret_15m": 0.001,
                "ret_1h": 0.004,
                "ret_4h": 0.012 if not high_crowding else 0.001,
                "ret_4h_percentile": 0.8 if not high_crowding else 0.5,
                "volume_z_1h": 1.4,
                "volume_z_4h": 0.9,
                "volume_1h_percentile": 0.8,
                "volume_4h_percentile": 0.75,
                "btc_ret_1h": 0.001,
                "btc_ret_4h": 0.003,
                "btc_market_state": "btc_up",
                "btc_vol_regime": "normal",
                "future_max_up_4h": future + 0.01,
                "future_max_down_4h": -0.005,
                "future_ret_4h": future / 2.0,
                "future_max_up_12h": future + 0.02,
                "future_max_down_12h": -0.008,
                "future_ret_12h": future,
            }
        )
    return rows


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    feature_path = tmp_path / "features.parquet"
    rows = []
    rows.extend(_feature_rows("BTCUSDT", 100.0, False))
    rows.extend(_feature_rows("AAAUSDT", 10.0, True))
    rows.extend(_feature_rows("BBBUSDT", 20.0, False))
    pd.DataFrame(rows).to_parquet(feature_path, index=False)
    trade_path = tmp_path / "trades.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "candidate": "CIC1_beta_extreme",
                "signal_time": "2026-01-01 02:00:00+00:00",
                "entry_time": "2026-01-01 02:15:00+00:00",
                "gross_return": 0.03,
                "net_return": 0.026,
                "base_signal_id": "AAA|1",
            },
            {
                "symbol": "BBBUSDT",
                "candidate": "CIC2_beta_broad",
                "signal_time": "2026-01-01 02:15:00+00:00",
                "entry_time": "2026-01-01 02:30:00+00:00",
                "gross_return": -0.01,
                "net_return": -0.014,
                "base_signal_id": "BBB|1",
            },
        ]
    ).to_csv(trade_path, index=False)
    return feature_path, trade_path


def test_v50_perp_crowding_atlas_outputs_core_tables(tmp_path: Path) -> None:
    feature_path, trade_path = _write_fixture(tmp_path)
    outputs = write_v50_perp_crowding_atlas(
        V50Config(
            report_root=tmp_path / "reports" / "v50",
            feature_path=feature_path,
            trade_cache_path=trade_path,
        )
    )

    for key in [
        "crowding_state_summary",
        "funding_oi_bucket_summary",
        "relative_value_pair_summary",
        "crowding_vs_cic_interaction",
        "crowding_action_atlas",
        "candidate_notes",
    ]:
        assert outputs[key].exists()

    state = pd.read_csv(outputs["crowding_state_summary"])
    assert "high_funding_high_oi" in set(state["state"])
    rv = pd.read_csv(outputs["relative_value_pair_summary"])
    assert "short_high_crowding_long_btc" in set(rv["candidate"])
    cic = pd.read_csv(outputs["crowding_vs_cic_interaction"])
    assert "candidate_x_crowding_state" in set(cic["bucket_type"])

