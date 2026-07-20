from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    V233Config,
    audit_v233_features,
    build_v233_event_features,
    load_v233_btc_15m,
    summarize_v233,
)


def test_v233_features_are_causal_and_cover_all_events() -> None:
    cfg = V233Config()
    events = __import__("pandas").read_parquet(cfg.event_path)
    for column in ("feature_time", "entry_time"):
        events[column] = __import__("pandas").to_datetime(events[column], utc=True)
    bars = load_v233_btc_15m(cfg)
    features = build_v233_event_features(events, bars, cfg)
    summary = summarize_v233(features)
    checks = audit_v233_features(events, bars, features, summary, cfg)
    assert checks["passed"].all()
    assert len(features) == len(events)
