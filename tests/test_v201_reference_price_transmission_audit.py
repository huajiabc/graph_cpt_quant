from pressure_graph.reports.v201_reference_price_transmission_audit import (
    parse_mapping,
)


def test_parse_mapping_round_trip_shape() -> None:
    parsed = parse_mapping("BTCUSDT:-0.5|ETHUSDT:0.25|SOLUSDT:0.25")
    assert parsed == {"BTCUSDT": -0.5, "ETHUSDT": 0.25, "SOLUSDT": 0.25}
