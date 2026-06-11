from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.config.v06a3 import load_v06a3_config
from pressure_graph.paper_live.v06a3 import _gate_audit, build_v06a3_paper_ledger


def _rows(symbol: str, rank30: int = 20, rank90: int = 20) -> list[dict]:
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
                "turnover_rank_30d": rank30,
                "turnover_rank_90d": rank90,
                "dynamic_all_rank": rank30,
                "dynamic_all_trailing_turnover": 1_000_000.0,
                "trailing_30d_turnover": 1_000_000.0,
                "core_liquidity": rank30 <= 30 and rank90 <= 50,
                "transient_hot": rank30 <= 30 and rank90 > 50,
                "ret_4h": 0.03 if idx == 0 else 0.0,
                "volume_z_4h": 2.5 if idx == 0 else 0.0,
                "warmup_complete": True,
                "symbol_volatility_percentile": 50,
            }
        )
    return rows


def test_v06a3_reclaim_lifecycle_records_order() -> None:
    cfg = load_v06a3_config()
    prepared = pd.DataFrame(_rows("AAAUSDT"))

    signals, trades, baseline_signals, baseline_trades = build_v06a3_paper_ledger(prepared, cfg)

    ir2_signals = signals[signals["candidate"].eq("IR2")]
    ir2_trades = trades[trades["candidate"].eq("IR2") & trades["portfolio_accepted"]]
    assert len(ir2_signals) == 1
    assert len(ir2_trades) == 1
    signal = ir2_signals.iloc[0]
    trade = ir2_trades.iloc[0]
    assert signal["status"] == "exited"
    assert pd.Timestamp(signal["pullback_time"]) == pd.Timestamp("2026-05-01 00:15:00Z")
    assert pd.Timestamp(signal["reclaim_time"]) == pd.Timestamp("2026-05-01 00:45:00Z")
    assert pd.Timestamp(signal["entry_time"]) == pd.Timestamp("2026-05-01 00:45:00Z")
    assert trade["exit_reason"] == "tp"
    assert trade["net_return_20bp"] == pytest.approx(0.036)
    assert trade["btc_state_at_signal"] == "BTC_up"
    assert trade["btc_state_at_entry"] == "BTC_up"
    audit = _gate_audit(signals, trades, prepared, cfg)
    primary_audit = audit[audit["is_primary"]]
    assert len(primary_audit) == 1
    assert primary_audit.iloc[0]["gate_passed"]
    assert not baseline_signals.empty
    assert isinstance(baseline_trades, pd.DataFrame)


def test_v06a3_transient_hot_veto_skips_candidate() -> None:
    cfg = load_v06a3_config()
    prepared = pd.DataFrame(_rows("AAAUSDT", rank30=10, rank90=80))

    signals, trades, _, _ = build_v06a3_paper_ledger(prepared, cfg)

    ir2 = signals[signals["candidate"].eq("IR2")].iloc[0]
    assert ir2["status"] == "skipped"
    assert ir2["skip_reason"] == "transient_hot"
    assert trades.empty or not trades["candidate"].eq("IR2").any()


def test_v06a3_primary_entry_requires_btc_up() -> None:
    cfg = load_v06a3_config()
    rows = _rows("AAAUSDT")
    rows[2]["btc_market_state"] = "BTC_chop"
    prepared = pd.DataFrame(rows)

    signals, trades, _, _ = build_v06a3_paper_ledger(prepared, cfg)

    ir2 = signals[signals["candidate"].eq("IR2")].iloc[0]
    assert ir2["status"] == "skipped"
    assert ir2["skip_reason"] == "entry_btc_not_up:BTC_chop"
    primary_ir2 = trades[
        trades["candidate"].eq("IR2")
        & trades["baseline_kind"].fillna("").eq("")
        & trades["portfolio_accepted"].fillna(False)
    ]
    assert primary_ir2.empty
    audit = _gate_audit(signals, trades, prepared, cfg)
    invalid = audit[audit["is_primary"]]
    assert len(invalid) == 1
    assert invalid.iloc[0]["btc_state_at_signal"] == "BTC_up"
    assert invalid.iloc[0]["btc_state_at_entry"] == "BTC_chop"
    assert not invalid.iloc[0]["gate_passed"]


def test_v06a3_exit_btc_chop_does_not_invalidate_existing_trade() -> None:
    cfg = load_v06a3_config()
    rows = _rows("AAAUSDT")
    rows[3]["btc_market_state"] = "BTC_chop"
    prepared = pd.DataFrame(rows)

    signals, trades, _, _ = build_v06a3_paper_ledger(prepared, cfg)

    ir2 = signals[signals["candidate"].eq("IR2")].iloc[0]
    primary_ir2 = trades[
        trades["candidate"].eq("IR2")
        & trades["baseline_kind"].fillna("").eq("")
        & trades["portfolio_accepted"].fillna(False)
    ]
    assert ir2["status"] == "exited"
    assert len(primary_ir2) == 1
    trade = primary_ir2.iloc[0]
    assert trade["btc_state_at_signal"] == "BTC_up"
    assert trade["btc_state_at_entry"] == "BTC_up"
    assert trade["btc_state_at_exit"] == "BTC_chop"
