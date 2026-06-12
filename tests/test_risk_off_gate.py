from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.live.risk_off_gate import (
    RiskOffGateConfig,
    annotate_signals_with_risk_off,
    build_risk_off_events,
    extract_long_signals,
    risk_off_decision,
    write_risk_off_shadow,
)


def _prepared_with_s5() -> pd.DataFrame:
    # 8 flat bars then breakdown/failed-bounce/lower-low so S5 fires (one symbol).
    close = [100, 100, 100, 100, 100, 100, 100, 100, 95, 94, 92, 91, 90, 90]
    n = len(close)
    times = pd.date_range("2026-02-01", periods=n, freq="15min", tz="UTC")
    close_arr = np.array(close, dtype=float)
    return pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": "AAAUSDT",
            "bar_open_time": times,
            "feature_time": times + pd.Timedelta(minutes=15),
            "open": np.concatenate([[close_arr[0]], close_arr[:-1]]),
            "high": close_arr * 1.001,
            "low": close_arr * 0.999,
            "close": close_arr,
            "dynamic_all_rank": 5,
            "gate_BTC_down": True,
            "bullish_volume_shock_event": False,
            "c2_mir1_raw": False,
            "c2_bucket_beta_extreme_overextended": False,
            "c2_beta_continuation": False,
        }
    )


def test_build_risk_off_events_detects_s5():
    events = build_risk_off_events(_prepared_with_s5(), RiskOffGateConfig(motifs=("S5",)))
    assert not events.empty
    assert set(events.columns) == {"symbol", "motif", "feature_time"}
    assert (events["motif"] == "S5").all()


def test_build_risk_off_events_missing_columns_is_safe():
    # A frame without the detector inputs should yield no events, not crash.
    frame = pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": "AAAUSDT",
            "bar_open_time": pd.date_range("2026-02-01", periods=5, freq="15min", tz="UTC"),
            "feature_time": pd.date_range("2026-02-01", periods=5, freq="15min", tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "dynamic_all_rank": 5,
        }
    )
    events = build_risk_off_events(frame)
    assert events.empty


def test_risk_off_decision_as_of():
    events = pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "motif": ["S1"],
            "feature_time": pd.to_datetime(["2026-02-01 08:00"], utc=True),
        }
    )
    cfg = RiskOffGateConfig(symbol_cooldown_bars=32)
    # 2h after the failure -> within 8h cooldown -> suppressed.
    suppressed, reason = risk_off_decision("AAAUSDT", pd.Timestamp("2026-02-01 10:00", tz="UTC"), events, cfg)
    assert suppressed is True and "recent_failure" in reason
    # Before the failure -> not suppressed (strict as-of).
    suppressed2, _ = risk_off_decision("AAAUSDT", pd.Timestamp("2026-02-01 07:00", tz="UTC"), events, cfg)
    assert suppressed2 is False
    # Different symbol -> not suppressed.
    suppressed3, _ = risk_off_decision("BBBUSDT", pd.Timestamp("2026-02-01 10:00", tz="UTC"), events, cfg)
    assert suppressed3 is False


def test_annotate_signals_with_risk_off():
    events = pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "motif": ["S1"],
            "feature_time": pd.to_datetime(["2026-02-01 08:00"], utc=True),
        }
    )
    signals = pd.DataFrame(
        {
            "exchange": ["bybit", "bybit"],
            "symbol": ["AAAUSDT", "AAAUSDT"],
            "candidate": ["CIC1_beta_extreme", "CIC1_beta_extreme"],
            "signal_time": pd.to_datetime(["2026-02-01 10:00", "2026-02-05 10:00"], utc=True),
        }
    )
    annotated = annotate_signals_with_risk_off(signals, events, RiskOffGateConfig(symbol_cooldown_bars=32))
    assert bool(annotated["risk_off_suppressed"].iloc[0]) is True
    assert bool(annotated["risk_off_suppressed"].iloc[1]) is False  # days later, outside cooldown


def test_extract_long_signals_from_gates():
    prepared = _prepared_with_s5().copy()
    prepared.loc[2, "bullish_volume_shock_event"] = True
    prepared.loc[2, "c2_bucket_beta_extreme_overextended"] = True
    signals = extract_long_signals(prepared)
    assert len(signals) == 1
    assert signals["candidate"].iloc[0] == "CIC1_beta_extreme"


def test_write_risk_off_shadow_outputs(tmp_path):
    prepared = _prepared_with_s5().copy()
    prepared.loc[12, "bullish_volume_shock_event"] = True
    prepared.loc[12, "c2_mir1_raw"] = True
    outputs = write_risk_off_shadow(prepared, tmp_path / "shadow", RiskOffGateConfig(motifs=("S5",)))
    for key in ["risk_off_events", "risk_off_signal_shadow", "risk_off_status"]:
        assert outputs[key].exists()
    status = outputs["risk_off_status"].read_text(encoding="utf-8")
    assert "shadow_only" in status
