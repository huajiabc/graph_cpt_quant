import numpy as np
import pandas as pd

from pressure_graph.reports.v225_alt_book_vacuum_pressure_to_btc import (
    V225Config,
    build_v225_no_vacuum_events,
    load_v225_inputs,
    price_v225_events,
)


def test_v225_priced_events_use_exact_four_hour_marks_and_costs() -> None:
    events, states, prices = load_v225_inputs()
    outcomes = price_v225_events(events, prices)
    assert len(outcomes) >= 150
    assert outcomes["exit_time"].sub(outcomes["entry_time"]).eq(
        pd.Timedelta(hours=4)
    ).all()
    assert np.allclose(
        outcomes["btc_primary_net_return_4h"],
        outcomes["btc_gross_return_4h"] - V225Config().btc_primary_cost,
    )
    assert np.allclose(
        outcomes["reversed_primary_net_return_4h"],
        -outcomes["btc_gross_return_4h"] - V225Config().btc_primary_cost,
    )
    no_vacuum = build_v225_no_vacuum_events(states)
    assert len(no_vacuum) == 307
    assert no_vacuum["withdrawing_symbol_count"].lt(5).all()
