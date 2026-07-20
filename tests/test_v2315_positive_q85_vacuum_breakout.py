from pressure_graph.reports.v2315_positive_q85_vacuum_breakout import (
    V2315Config,
    _decision_config,
)


def test_v2315_decision_config_freezes_q85_and_trigger_counts() -> None:
    cfg = _decision_config(V2315Config())
    assert cfg.pressure_quantile == 0.85
    assert cfg.minimum_total_triggers == 70
    assert cfg.minimum_period_triggers == 20
