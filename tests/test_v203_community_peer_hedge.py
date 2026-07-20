from pressure_graph.reports.v203_community_peer_hedge import parse_weights


def test_parse_weights() -> None:
    assert parse_weights("A:-0.25|B:-0.25|C:0.5") == {
        "A": -0.25,
        "B": -0.25,
        "C": 0.5,
    }
