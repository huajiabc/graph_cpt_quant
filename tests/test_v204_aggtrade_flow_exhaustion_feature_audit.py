import pandas as pd

from pressure_graph.reports.v204_aggtrade_flow_exhaustion_feature_audit import (
    EVENT_WIDE,
    STATE_SPREAD,
    add_v204_quality_fields,
    build_v204_candidate_features,
)


def _leg(
    event: str,
    symbol: str,
    early: float,
    late: float,
    source_sign: float = 1.0,
) -> dict[str, object]:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    return {
        "task_id": f"{event}|{symbol}",
        "source_event_id": event,
        "feature_time": start + pd.Timedelta(minutes=15),
        "window_start": start,
        "window_end": start + pd.Timedelta(minutes=15),
        "first_trade_time": start + pd.Timedelta(seconds=1),
        "last_trade_time": start + pd.Timedelta(minutes=14, seconds=59),
        "period": "validation",
        "community_id": "C1",
        "source_sign": source_sign,
        "symbol": symbol,
        "trade_count": 200,
        "price_return": 0.01,
        "aligned_price_return": 0.01,
        "buy_sell_imbalance": 0.1,
        "aligned_buy_sell_imbalance": 0.1,
        "early_aligned_imbalance": early,
        "middle_aligned_imbalance": 0.0,
        "late_aligned_imbalance": late,
        "aligned_flow_exhaustion": early - late,
        "late_flow_opposes_source": late < 0,
        "large_aligned_imbalance": 0.1,
    }


def test_quality_fields_classify_strict_exhaustion() -> None:
    frame = add_v204_quality_fields(pd.DataFrame([_leg("E1", "A", 0.4, -0.2)]))
    assert frame.loc[0, "quality_ok"]
    assert frame.loc[0, "strict_exhausted"]
    assert not frame.loc[0, "persistent_flow"]


def test_candidate_builder_uses_frozen_sign_rules() -> None:
    rows = [
        _leg("E1", "A", 0.4, -0.2),
        _leg("E1", "B", 0.3, -0.1),
        _leg("E1", "C", 0.2, 0.1),
        _leg("E1", "D", 0.2, 0.2),
    ]
    candidates = build_v204_candidate_features(
        add_v204_quality_fields(pd.DataFrame(rows))
    )
    assert set(candidates["candidate"]) == {EVENT_WIDE, STATE_SPREAD}
    wide = candidates[candidates["candidate"].eq(EVENT_WIDE)].iloc[0]
    spread = candidates[candidates["candidate"].eq(STATE_SPREAD)].iloc[0]
    assert wide["late_opposes_fraction"] == 0.5
    assert spread["strict_exhausted_receivers"] == "A|B"
    assert spread["persistent_receivers"] == "C|D"


def test_candidate_builder_rejects_insufficient_persistent_bucket() -> None:
    rows = [
        _leg("E1", "A", 0.4, -0.2),
        _leg("E1", "B", -0.2, 0.1),
        _leg("E1", "C", -0.1, 0.2),
    ]
    candidates = build_v204_candidate_features(
        add_v204_quality_fields(pd.DataFrame(rows))
    )
    assert candidates.empty or STATE_SPREAD not in set(candidates["candidate"])
