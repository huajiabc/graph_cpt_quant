from pressure_graph.reports.v206_aggtrade_flow_exhaustion_audit import parse_mapping


def test_parse_mapping_round_trip_format() -> None:
    assert parse_mapping("A:0.25|BTCUSDT:-0.125") == {
        "A": 0.25,
        "BTCUSDT": -0.125,
    }
    assert parse_mapping("") == {}
