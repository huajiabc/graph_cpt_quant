from pressure_graph.reports.v2341_liquidation_graph_bucket_independent_audit import (
    run_v2341,
)


def test_v2341_graph_bucket_survives_source_and_receiver_leave_one_out() -> None:
    audit, source_loo, receiver_loo, metadata = run_v2341()
    assert audit["passed"].all(), audit.loc[~audit["passed"]].to_dict("records")
    assert len(source_loo) == 17
    assert len(receiver_loo) == 17
    assert metadata["promotion_allowed"] is False
