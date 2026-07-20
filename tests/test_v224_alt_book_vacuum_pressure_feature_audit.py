import numpy as np
import pandas as pd

from pressure_graph.reports.v224_alt_book_vacuum_pressure_feature_audit import (
    V224Config,
    add_v224_symbol_states,
    audit_v224_features,
    build_v224_bucket_states,
    select_v224_events,
    summarize_v224_events,
)
from pressure_graph.reports.v161_hourly_liquidity_withdrawal_amplification import (
    load_v161_features,
)


def test_v224_feature_events_are_causal_and_meet_frozen_breadth() -> None:
    cfg = V224Config()
    states = add_v224_symbol_states(load_v161_features(), cfg)
    buckets = build_v224_bucket_states(states, cfg)
    events = select_v224_events(buckets, cfg)
    summary = summarize_v224_events(events)
    checks = audit_v224_features(states, buckets, events, summary, cfg)
    assert checks["passed"].all()
    assert events["directional_symbol_count"].ge(11).all()
    assert events["withdrawing_symbol_count"].ge(5).all()
    assert events["signal_direction"].eq(np.sign(events["bucket_pressure"])).all()
    assert events["entry_time"].sort_values().diff().dropna().ge(
        pd.Timedelta(hours=4)
    ).all()
