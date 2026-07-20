import pandas as pd

from pressure_graph.reports.v227_vacuum_pressure_cross_section_feature_audit import (
    V227Config,
    audit_v227_features,
    build_v227_rank_features,
    summarize_v227,
)


def test_v227_rank_features_form_exact_top4_bottom4_spread() -> None:
    events = pd.read_parquet(V227Config().event_path)
    states = pd.read_parquet(V227Config().symbol_state_path)
    for frame, column in ((events, "entry_time"), (states, "decision_time")):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    features = build_v227_rank_features(events, states)
    summary = summarize_v227(features)
    checks = audit_v227_features(features, summary)
    assert checks["passed"].all()
    counts = features.groupby(["entry_time", "side"])["symbol"].nunique()
    assert counts.eq(4).all()
