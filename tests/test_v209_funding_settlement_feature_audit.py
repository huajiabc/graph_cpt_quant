import pandas as pd

from pressure_graph.reports.v209_funding_settlement_feature_audit import (
    ALL_NEGATIVE,
    NEW_NEGATIVE,
    V209FeatureConfig,
    build_v209_candidate_features,
    build_v209_funding_features,
)


def test_funding_feature_builder_and_candidate_rules() -> None:
    symbols = {f"S{i}" for i in range(5)}
    rows = []
    for symbol in sorted(symbols):
        rows.extend(
            [
                {
                    "symbol": symbol,
                    "funding_time": pd.Timestamp("2025-07-31 16:00", tz="UTC"),
                    "funding_rate_settled": 0.0001,
                },
                {
                    "symbol": symbol,
                    "funding_time": pd.Timestamp("2025-08-01 00:00", tz="UTC"),
                    "funding_rate_settled": -0.0002,
                },
            ]
        )
    cfg = V209FeatureConfig(
        expected_symbols=5,
        minimum_synchronized_coverage=5,
        minimum_candidate_events=1,
        minimum_period_events=0,
    )
    features = build_v209_funding_features(pd.DataFrame(rows), symbols, cfg)
    entry = pd.Timestamp("2025-08-01 00:15", tz="UTC")
    exit_time = pd.Timestamp("2025-08-01 01:15", tz="UTC")
    close = pd.DataFrame(
        1.0,
        index=[entry, exit_time],
        columns=["BTCUSDT", *sorted(symbols)],
    )
    candidates = build_v209_candidate_features(features, close, cfg)
    assert set(candidates["candidate"]) == {ALL_NEGATIVE, NEW_NEGATIVE}
    assert candidates["selection_count"].eq(5).all()
    assert candidates["entry_time"].sub(candidates["settlement_time"]).eq(
        pd.Timedelta(minutes=15)
    ).all()
