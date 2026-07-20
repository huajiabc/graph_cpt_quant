from pressure_graph.reports.v148_funding_sign_hysteresis import update_sign_states


def test_v148_requires_two_consecutive_opposite_signs() -> None:
    states, streaks = update_sign_states({}, {}, {"A": -1.0}, 2)
    assert states["A"] == -1
    states, streaks = update_sign_states(states, streaks, {"A": 1.0}, 2)
    assert states["A"] == -1
    assert streaks["A"] == 1
    states, streaks = update_sign_states(states, streaks, {"A": 2.0}, 2)
    assert states["A"] == 1
    assert streaks["A"] == 0


def test_v148_same_sign_resets_pending_flip() -> None:
    states, streaks = update_sign_states({"A": -1}, {"A": 1}, {"A": -0.1}, 2)
    assert states["A"] == -1
    assert streaks["A"] == 0


def test_v148_missing_symbol_exits_immediately() -> None:
    states, streaks = update_sign_states(
        {"A": -1, "B": 1}, {"A": 0, "B": 0}, {"B": 1.0}, 2
    )
    assert "A" not in states
    assert "A" not in streaks
