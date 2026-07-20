from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.paper_live.v07d2 import _add_btc_risk_context
from pressure_graph.reports.v91_p2_portfolio_risk_shadows import (
    P2_BETA,
    P2_CORR,
    P2_EW,
    P2_VOL,
    build_p2_portfolio_risk_shadows,
)


def _pool() -> pd.DataFrame:
    base = pd.Timestamp("2026-07-01T00:00:00Z")
    rows = []
    for idx in range(10):
        rows.append(
            {
                "trade_id": f"t{idx}",
                "signal_id": f"s{idx}",
                "shadow_base_signal_id": f"base{idx}",
                "exchange": "bybit",
                "symbol": f"S{idx}USDT",
                "candidate": "CIC1_FILTERED_MIR1" if idx % 2 == 0 else "CIC2_FILTERED_MIR1",
                "candidate_priority": 2 if idx % 2 == 0 else 1,
                "entry_time": base + pd.Timedelta(minutes=idx),
                "exit_time": base + pd.Timedelta(hours=4),
                "entry_price": 100.0,
                "sl_price": 98.0 if idx % 2 == 0 else 97.0,
                "btc_beta_7d_at_entry": 2.0,
                "btc_corr_7d_at_entry": 0.9,
                "cluster_id_at_entry": "cluster-a" if idx < 4 else f"cluster-{idx}",
                "burst_id": "b1",
                "local_volume_shock_time": base,
                "net_return_10bp": 0.01,
                "net_return_20bp": 0.008,
                "net_return_30bp": 0.006,
            }
        )
    return pd.DataFrame(rows)


def test_four_risk_shadow_arms_enforce_frozen_constraints() -> None:
    selected, skipped, summary = build_p2_portfolio_risk_shadows(_pool())
    counts = selected.groupby("risk_shadow_arm").size().to_dict()
    assert counts[P2_EW] == 8
    assert counts[P2_VOL] >= 6
    assert counts[P2_BETA] == 3
    assert counts[P2_CORR] == 8

    beta = selected[selected["risk_shadow_arm"].eq(P2_BETA)]
    assert beta["btc_beta_exposure_after"].max() <= 6.0
    corr = selected[selected["risk_shadow_arm"].eq(P2_CORR)]
    assert (corr["correlation_cluster"].eq("cluster-a")).sum() == 2
    assert "correlation_cluster_cap" in set(skipped["skip_reason"])
    assert set(summary["risk_shadow_arm"]) == {P2_EW, P2_VOL, P2_BETA, P2_CORR}


def test_btc_risk_context_uses_only_rolling_past_rows() -> None:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for symbol, multiplier in [("BTCUSDT", 1.0), ("AAAUSDT", 2.0)]:
        for idx in range(220):
            ret = ((idx % 7) - 3) / 1000.0
            rows.append(
                {
                    "exchange": "bybit",
                    "symbol": symbol,
                    "bar_open_time": base + pd.Timedelta(minutes=15 * idx),
                    "ret_1h": ret * multiplier,
                    "btc_ret_1h": ret,
                }
            )
    frame = pd.DataFrame(rows)
    enriched = _add_btc_risk_context(frame, window_bars=200, min_periods=192)
    aaa = enriched[enriched["symbol"].eq("AAAUSDT")].sort_values("bar_open_time")
    assert pd.isna(aaa.iloc[190]["btc_beta_7d"])
    assert aaa.iloc[-1]["btc_beta_7d"] == pytest.approx(2.0, rel=1e-6)
    assert aaa.iloc[-1]["btc_corr_7d"] == pytest.approx(1.0, rel=1e-6)
