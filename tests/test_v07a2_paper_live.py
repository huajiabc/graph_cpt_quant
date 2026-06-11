from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.config.v07a2 import load_v07a2_config
from pressure_graph.paper_live.v07a2 import _market_gate_audit, build_v07a2_paper_ledger


def _rows(symbol: str) -> list[dict]:
    times = pd.date_range("2026-05-01 00:00:00Z", periods=6, freq="15min", tz="UTC")
    lows = [100.0, 98.8, 99.1, 100.1, 100.0, 100.0]
    closes = [100.0, 99.1, 100.4, 100.5, 101.0, 101.0]
    highs = [100.2, 100.0, 100.5, 105.0, 101.2, 101.2]
    rows = []
    for idx, ts in enumerate(times):
        rows.append(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "bar_open_time": ts,
                "bar_close_time": ts + pd.Timedelta(minutes=15),
                "feature_time": ts + pd.Timedelta(minutes=15),
                "open": 100.5 if idx == 3 else 100.0,
                "high": highs[idx],
                "low": lows[idx],
                "close": closes[idx],
                "funding_time": ts,
                "funding_rate_settled": 0.0,
                "btc_market_state": "BTC_up",
                "btc_ret_1h": 0.01,
                "btc_ret_4h": 0.02,
                "btc_volatility_4h": 0.01,
                "dynamic_all_rank": 20,
                "dynamic_all_trailing_turnover": 1_000_000.0,
                "turnover_rank_90d": 20,
                "volume_impulse_density": 0.12,
                "market_volume_impulse_density_high": True,
                "market_low_volume_impulse_density": False,
                "market_btc_up": True,
                "market_btc_chop": False,
                "ret_4h": 0.03 if idx == 0 else 0.0,
                "volume_z_4h": 2.5 if idx == 0 else 0.0,
                "warmup_complete": True,
                "symbol_volatility_percentile": 50,
            }
        )
    return rows


def test_v07a2_mir1_lifecycle_records_market_graph_context() -> None:
    cfg = load_v07a2_config()
    prepared = pd.DataFrame(_rows("AAAUSDT"))

    signals, trades, baseline_signals, baseline_trades = build_v07a2_paper_ledger(prepared, cfg)

    mir1_signals = signals[signals["candidate"].eq("MIR1")]
    mir1_trades = trades[trades["candidate"].eq("MIR1") & trades["portfolio_accepted"]]
    assert len(mir1_signals) == 1
    assert len(mir1_trades) == 1
    signal = mir1_signals.iloc[0]
    trade = mir1_trades.iloc[0]
    assert signal["status"] == "exited"
    assert pd.Timestamp(signal["pullback_time"]) == pd.Timestamp("2026-05-01 00:15:00Z")
    assert pd.Timestamp(signal["reclaim_time"]) == pd.Timestamp("2026-05-01 00:45:00Z")
    assert pd.Timestamp(signal["entry_time"]) == pd.Timestamp("2026-05-01 00:45:00Z")
    assert trade["exit_reason"] == "tp"
    assert trade["net_return_20bp"] == pytest.approx(0.036)
    assert trade["market_gate_at_signal"]
    assert trade["market_gate_at_entry"]
    assert trade["volume_impulse_density_at_signal"] == pytest.approx(0.12)
    audit = _market_gate_audit(signals, trades, prepared)
    primary_audit = audit[audit["is_primary"]]
    assert len(primary_audit) == 1
    assert primary_audit.iloc[0]["gate_passed"]
    assert not baseline_signals.empty
    assert isinstance(baseline_trades, pd.DataFrame)


def test_v07a2_primary_entry_requires_market_gate() -> None:
    cfg = load_v07a2_config()
    rows = _rows("AAAUSDT")
    rows[2]["volume_impulse_density"] = 0.01
    rows[2]["market_volume_impulse_density_high"] = False
    prepared = pd.DataFrame(rows)

    signals, trades, _, _ = build_v07a2_paper_ledger(prepared, cfg)

    mir1 = signals[signals["candidate"].eq("MIR1")].iloc[0]
    assert mir1["status"] == "skipped"
    assert mir1["skip_reason"] == "entry_market_gate_off:market_volume_impulse_density_high"
    primary = trades[trades["candidate"].eq("MIR1") & trades["portfolio_accepted"].fillna(False)]
    assert primary.empty
    audit = _market_gate_audit(signals, trades, prepared)
    invalid = audit[audit["is_primary"]]
    assert len(invalid) == 1
    assert invalid.iloc[0]["market_gate_at_signal"]
    assert not invalid.iloc[0]["market_gate_at_entry"]
    assert not invalid.iloc[0]["gate_passed"]


def test_v07a2_exit_market_gate_off_does_not_invalidate_existing_trade() -> None:
    cfg = load_v07a2_config()
    rows = _rows("AAAUSDT")
    rows[3]["volume_impulse_density"] = 0.01
    rows[3]["market_volume_impulse_density_high"] = False
    prepared = pd.DataFrame(rows)

    signals, trades, _, _ = build_v07a2_paper_ledger(prepared, cfg)

    mir1 = signals[signals["candidate"].eq("MIR1")].iloc[0]
    primary = trades[trades["candidate"].eq("MIR1") & trades["portfolio_accepted"].fillna(False)]
    assert mir1["status"] == "exited"
    assert len(primary) == 1
    trade = primary.iloc[0]
    assert trade["market_gate_at_signal"]
    assert trade["market_gate_at_entry"]
    assert not trade["market_gate_at_exit"]
