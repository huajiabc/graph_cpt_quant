from pressure_graph.reports.v2340_liquidation_graph_volatility_transmission_pilot import (
    run_v2340,
)


def test_v2340_graph_bucket_beats_single_sources_but_is_not_promotable() -> None:
    result = run_v2340()
    assert result["audit"]["passed"].all(), result["audit"].loc[
        ~result["audit"]["passed"]
    ].to_dict("records")
    assert len(result["matrix"]) == 289
    assert result["metadata"]["promotion_allowed"] is False
    assert result["metadata"]["single_source_selection_allowed"] is False
