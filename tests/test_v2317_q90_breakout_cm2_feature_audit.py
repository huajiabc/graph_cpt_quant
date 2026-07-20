import pandas as pd

from pressure_graph.reports.v2317_q90_breakout_cm2_feature_audit import (
    V2317Config,
    audit_v2317,
    build_v2317_event_mapping,
    summarize_v2317,
)


def test_v2317_maps_events_without_outcomes() -> None:
    times = pd.date_range("2026-01-05", periods=49, freq="7D", tz="UTC")
    periods = ["development"] * 17 + ["validation"] * 16 + ["holdout"] * 16
    calendar = pd.DataFrame(
        {
            "entry_time": times,
            "exit_time": times + pd.Timedelta(days=7),
            "month_start": pd.DatetimeIndex(
                [pd.Timestamp(time.year, time.month, 1, tz="UTC") for time in times]
            ),
            "period": periods,
            "candidate": "CM2",
        }
    )
    event_times = []
    event_periods = []
    for index, week in calendar.iloc[:17].iterrows():
        count = 2 if index < 2 else 1
        for offset in range(count):
            event_times.append(week.entry_time + pd.Timedelta(hours=4 + 8 * offset))
            event_periods.append(week.period)
    for _, week in calendar.iloc[17:33].iterrows():
        event_times.append(week.entry_time + pd.Timedelta(hours=4))
        event_periods.append(week.period)
    for _, week in calendar.iloc[33:49].iterrows():
        event_times.append(week.entry_time + pd.Timedelta(hours=4))
        event_periods.append(week.period)
    features = pd.DataFrame(
        {
            "candidate": "feature",
            "feature_time": event_times,
            "entry_time": event_times,
            "entry_month": [time.strftime("%Y-%m") for time in event_times],
            "period": event_periods,
            "signal_direction": 1,
            "bucket_pressure": 1.0,
            "prior_abs_pressure_threshold": 0.9,
            "covered_symbols": 16,
            "directional_symbol_count": 12,
            "directional_breadth": 0.75,
            "withdrawing_symbol_count": 5,
            "withdrawal_breadth": 0.3125,
        }
    )
    cfg = V2317Config(minimum_active_months=1)
    mapping = build_v2317_event_mapping(features, calendar, cfg)
    summary = summarize_v2317(mapping, calendar)
    checks = audit_v2317(mapping, calendar, summary, cfg)
    assert len(mapping) == 51
    assert checks["passed"].all()
    assert "primary_net_return" not in mapping.columns
