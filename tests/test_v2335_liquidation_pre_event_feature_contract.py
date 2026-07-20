import pandas as pd

from pressure_graph.reports.v2335_liquidation_pre_event_feature_contract import (
    build_v2335_causal_features,
    write_v2335_contract,
)


def _event(
    event_time: str,
    first_seen_at: str,
    side: str,
    liquidation_side: str,
    notional: float,
    symbol: str = "BTCUSDT",
) -> dict[str, object]:
    return {
        "event_time": pd.Timestamp(event_time),
        "first_seen_at": pd.Timestamp(first_seen_at),
        "position_side": side,
        "liquidation_side": liquidation_side,
        "notional_usd": notional,
        "bybit_symbol": symbol,
    }


def test_v2335_excludes_future_and_late_known_events() -> None:
    decision = pd.Timestamp("2026-07-17T05:00:00Z")
    events = pd.DataFrame(
        [
            _event(
                "2026-07-17T04:50:00Z",
                "2026-07-17T04:51:00Z",
                "long",
                "sell",
                100.0,
            ),
            _event(
                "2026-07-17T04:55:00Z",
                "2026-07-17T05:01:00Z",
                "short",
                "buy",
                200.0,
            ),
            _event(
                "2026-07-17T05:00:00Z",
                "2026-07-17T05:00:00Z",
                "short",
                "buy",
                300.0,
            ),
            _event(
                "2026-07-17T05:01:00Z",
                "2026-07-17T05:01:00Z",
                "short",
                "buy",
                400.0,
            ),
        ]
    )
    features = build_v2335_causal_features(events, [decision])
    assert features.loc[0, "liq_15m_event_count"] == 1
    assert features.loc[0, "liq_15m_total_usd"] == 100.0
    assert features.loc[0, "liq_15m_forced_sell_usd"] == 100.0
    assert features.loc[0, "liq_15m_forced_buy_usd"] == 0.0


def test_v2335_current_contract_writes_without_outcomes(tmp_path) -> None:
    paths = write_v2335_contract(
        report_root=tmp_path / "report",
        prereg_path=tmp_path / "prereg.md",
    )
    checks = pd.read_csv(paths["checks"])
    features = pd.read_parquet(paths["feature_snapshot"])
    assert checks["passed"].all()
    assert not any("return" in column for column in features.columns)
