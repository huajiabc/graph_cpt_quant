from pathlib import Path

import pandas as pd
import pytest

from pressure_graph.reports.v107_flow_graph_readiness import (
    V107Config,
    build_v107_synchronized_panel,
    evaluate_v107_readiness,
)


def _bars(start: pd.Timestamp, minutes: int, symbols: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for timestamp in pd.date_range(start, periods=minutes, freq="1min"):
        for symbol in symbols:
            for exchange in ("binance", "bybit"):
                rows.append(
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "bar_open_time": timestamp,
                        "buy_sell_imbalance": 0.2 if exchange == "binance" else 0.1,
                        "turnover": 1000.0,
                        "close": 100.0,
                        "event_lag_seconds": 2.0,
                        "bar_complete": True,
                    }
                )
    return pd.DataFrame(rows)


def test_v107_requires_both_venues_and_builds_flow_fields() -> None:
    start = pd.Timestamp("2026-07-13T11:01:00Z")
    bars = _bars(start, 3, ("A",))
    bars = bars[~(
        bars["exchange"].eq("bybit")
        & bars["bar_open_time"].eq(start + pd.Timedelta(minutes=2))
    )]
    cfg = V107Config(frozen_symbols=("A",), admissible_start=start)
    panel = build_v107_synchronized_panel(bars, cfg)
    assert len(panel) == 2
    assert panel["cross_venue_imbalance"].iloc[0] == pytest.approx(0.15)
    assert bool(panel["flow_sign_agreement"].all())


def test_v107_gate_opens_only_after_time_and_quality_requirements() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    symbols = ("A", "B")
    panel = build_v107_synchronized_panel(
        _bars(start, 3, symbols),
        V107Config(frozen_symbols=symbols, admissible_start=start),
    )
    cfg = V107Config(
        tape_root=Path("unused"),
        frozen_symbols=symbols,
        admissible_start=start,
        required_calendar_days=0,
        required_months=1,
        required_usable_symbols=2,
        required_full_days_per_symbol=1,
        minimum_evaluated_day_minutes=3,
    )
    readiness, coverage = evaluate_v107_readiness(
        panel, cfg, as_of=start + pd.Timedelta(minutes=3)
    )
    assert len(coverage) == 2
    assert readiness.iloc[0]["status"] == "RESEARCH_GATE_OPEN"
    assert bool(readiness.iloc[0]["alpha_verdict_allowed"])


def test_v107_missing_local_tape_never_emits_alpha_verdict() -> None:
    readiness, _ = evaluate_v107_readiness(pd.DataFrame(), V107Config())
    assert readiness.iloc[0]["status"] == "DATA_UNAVAILABLE_LOCAL"
    assert not bool(readiness.iloc[0]["alpha_verdict_allowed"])
