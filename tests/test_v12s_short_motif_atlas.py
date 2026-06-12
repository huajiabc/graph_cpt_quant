from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pressure_graph.reports.v12s_short_motif_atlas as atlas
from pressure_graph.reports.v12s_short_motif_atlas import (
    MOTIF_PARAMS,
    ShortAtlasConfig,
    detect_s1_failed_reclaim,
    detect_s3_crowded_long_unwind,
    detect_s5_btc_down_breakdown,
    write_v12s_short_motif_atlas,
)


def _ohlc(close: list[float]) -> pd.DataFrame:
    close_arr = np.array(close, dtype=float)
    return pd.DataFrame(
        {
            "open": np.concatenate([[close_arr[0]], close_arr[:-1]]),
            "high": close_arr * 1.001,
            "low": close_arr * 0.999,
            "close": close_arr,
        }
    )


def test_detect_s1_failed_reclaim_fires_on_pattern():
    # shock at 0 (close=100), pullback to 98.5 (bar2), reclaim >=100 (bar3),
    # then fail close <100 (bar4).
    close = [100.0, 99.0, 98.5, 100.5, 99.0, 98.0]
    group = _ohlc(close)
    group["low"] = [100, 99, 98.5, 100, 99, 98.0]  # ensure pullback dips below 99
    group["bullish_volume_shock_event"] = [True, False, False, False, False, False]
    signals = detect_s1_failed_reclaim(group, MOTIF_PARAMS["S1"])
    assert signals, "S1 should detect a failed reclaim"
    confirmation_idx, anchor_idx = signals[0]
    assert anchor_idx == 0
    assert confirmation_idx == 4


def test_detect_s5_btc_down_breakdown_fires():
    # 8 flat bars, then breakdown (95), failed bounce (red, lower high at 94),
    # then a lower low (92) -> confirmation.
    close = [100, 100, 100, 100, 100, 100, 100, 100, 95, 94, 92]
    group = _ohlc(close)
    group["gate_BTC_down"] = [True] * len(close)
    signals = detect_s5_btc_down_breakdown(group, MOTIF_PARAMS["S5"])
    assert signals, "S5 should detect a breakdown + failed bounce + lower low"
    confirmation_idx, anchor_idx = signals[0]
    assert anchor_idx == 8 and confirmation_idx == 10


def test_detect_s3_requires_crowded_state():
    n = 12
    group = _ohlc([100.0] * n)
    group["funding_percentile"] = [90.0] * n
    group["oi_value_delta_4h_percentile"] = [85.0] * n
    group["ret_4h_percentile"] = [40.0] * n
    group["gate_BTC_up"] = [False] * n
    # support break: close drops below recent min after the crowded anchor
    group.loc[10, "close"] = 90.0
    signals = detect_s3_crowded_long_unwind(group, MOTIF_PARAMS["S3"])
    assert signals, "S3 should fire when crowded + stalled + support break"


def _synthetic_symbol(symbol: str, base: float, btc_state: str) -> pd.DataFrame:
    n = 240
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    close = base * np.exp(np.cumsum(rng.normal(-0.0003, 0.01, n)))
    frame = pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": symbol,
            "bar_open_time": times,
            "bar_close_time": times + pd.Timedelta(minutes=15),
            "feature_time": times + pd.Timedelta(minutes=15),
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "dynamic_all_rank": 5,
            "warmup_complete": True,
            "btc_market_state": btc_state,
            "gate_BTC_up": btc_state == "BTC_up",
            "gate_BTC_chop": btc_state == "BTC_chop",
            "gate_BTC_down": btc_state == "BTC_down",
            "ret_4h_percentile": rng.uniform(0, 100, n),
            "volume_z_4h": rng.uniform(0, 4, n),
            "upper_wick_ratio": rng.uniform(0, 1, n),
            "funding_percentile": rng.uniform(0, 100, n),
            "oi_value_delta_4h_percentile": rng.uniform(0, 100, n),
        }
    )
    # Guarantee some shocks so S1 has anchors.
    frame["bullish_volume_shock_event"] = False
    frame.loc[[20, 80, 140, 200], "bullish_volume_shock_event"] = True
    return frame


def test_write_v12s_short_motif_atlas_end_to_end(monkeypatch, tmp_path):
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    states = {"AAAUSDT": "BTC_down", "BBBUSDT": "BTC_chop", "CCCUSDT": "BTC_up"}
    frames = {s: _synthetic_symbol(s, 100.0 + i * 10, states[s]) for i, s in enumerate(symbols)}

    rank30 = pd.DataFrame({"symbol": symbols, "dynamic_all_rank": [5, 6, 7], "month_start": pd.Timestamp("2026-01-01", tz="UTC")})

    def fake_rank_inputs(feature_path, instruments, config):
        return rank30, rank30.copy(), symbols

    def fake_read_symbol(feature_path, r30, r90, symbol, config):
        return frames[symbol].copy()

    monkeypatch.setattr(atlas, "_rank_inputs", fake_rank_inputs)
    monkeypatch.setattr(atlas, "_read_symbol_features", fake_read_symbol)

    cfg = ShortAtlasConfig(report_root=tmp_path / "atlas")
    outputs = write_v12s_short_motif_atlas(Path("ignored.parquet"), pd.DataFrame(), None, cfg)

    for key in [
        "candidate_summary",
        "baseline_comparison",
        "regime_split",
        "month_cap",
        "symbol_contribution",
        "candidate_notes",
    ]:
        assert outputs[key].exists(), key

    summary = pd.read_csv(outputs["candidate_summary"])
    if not summary.empty:
        assert {"motif", "cost_single_side_bps", "short_net", "squeeze_out_rate"}.issubset(summary.columns)
        # cost grid present
        assert set(summary["cost_single_side_bps"].unique()).issubset({10.0, 20.0, 30.0, 50.0})

    baselines = pd.read_csv(outputs["baseline_comparison"])
    assert {"real_net20", "entry_only_net20", "matched_random_net20", "plain_drop_net20"}.issubset(baselines.columns)

    notes = outputs["candidate_notes"].read_text(encoding="utf-8")
    assert "Short Motif Atlas" in notes
    assert "just-not-being-long" in notes


def test_matched_random_is_deterministic(monkeypatch):
    frame = _synthetic_symbol("AAAUSDT", 100.0, "BTC_chop")
    cfg = ShortAtlasConfig()
    first = atlas._matched_random_signals(frame, cfg, "S1", 5)
    second = atlas._matched_random_signals(frame, cfg, "S1", 5)
    assert first == second
