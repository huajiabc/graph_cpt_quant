from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v51_funding_extreme_oi_low_rv_audit import (
    V51Config,
    write_v51_funding_extreme_oi_low_rv_audit,
)


def _rows(symbol: str, base: float, *, candidate: bool, future: float, funding: float) -> list[dict[str, object]]:
    times = pd.date_range("2026-01-01", periods=130, freq="15min", tz="UTC")
    out: list[dict[str, object]] = []
    for idx, ts in enumerate(times):
        close = base * (1.0 + idx * 0.0002)
        out.append(
            {
                "symbol": symbol,
                "feature_time": ts,
                "bar_close_time": ts,
                "close": close,
                "warmup_complete": True,
                "universe_dynamic_monthly_top30": True,
                "funding_time": ts.floor("8h"),
                "funding_rate_settled": funding,
                "funding_interval_minutes": 480.0,
                "funding_age_minutes": float((ts - ts.floor("8h")).total_seconds() / 60.0),
                "funding_percentile": 95.0 if candidate else 50.0,
                "oi_value_delta_4h_percentile": 10.0 if candidate else 55.0,
                "oi_delta_4h_percentile": 10.0 if candidate else 55.0,
                "ret_4h": 0.01,
                "ret_4h_percentile": 70.0,
                "btc_market_state": "btc_up",
                "btc_ret_4h": 0.01,
                "future_ret_4h": future / 3.0,
                "future_ret_12h": future,
            }
        )
    return out


def test_v51_funding_extreme_oi_low_audit_outputs_core_tables(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.parquet"
    rows = []
    rows.extend(_rows("BTCUSDT", 100.0, candidate=False, future=0.005, funding=0.0001))
    rows.extend(_rows("ETHUSDT", 50.0, candidate=False, future=0.006, funding=0.00012))
    rows.extend(_rows("AAAUSDT", 10.0, candidate=True, future=-0.004, funding=0.0010))
    rows.extend(_rows("BBBUSDT", 20.0, candidate=False, future=0.003, funding=0.00005))
    pd.DataFrame(rows).to_parquet(feature_path, index=False)

    outputs = write_v51_funding_extreme_oi_low_rv_audit(
        V51Config(report_root=tmp_path / "reports" / "v51", feature_path=feature_path)
    )

    for key in [
        "rv_candidate_ledger",
        "price_funding_pnl_decomposition",
        "time_to_funding_bucket_summary",
        "hedge_leg_comparison",
        "month_symbol_stability",
        "matched_random_baseline",
        "cost_stress",
        "candidate_notes",
    ]:
        assert outputs[key].exists()

    decomposition = pd.read_csv(outputs["price_funding_pnl_decomposition"])
    assert "BTC" in set(decomposition["hedge_leg"])
    ledger = pd.read_csv(outputs["rv_candidate_ledger"])
    assert set(ledger["hedge_leg"]).issuperset({"BTC", "ETH", "BTC_ETH", "MARKET"})

