from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v90_p2_continuation_strength import (
    P2_PORTFOLIO_ID,
    add_continuation_strength,
    burst_level,
    prepare_p2_core,
)


def _rows() -> pd.DataFrame:
    base = pd.Timestamp("2026-07-10T00:00:00Z")
    rows = []
    for idx, (density, beta, shock, reclaim_minutes, burst, net20) in enumerate(
        [
            (0.10, 80.0, 2.0, 120, "b1", -0.01),
            (0.20, 90.0, 4.0, 60, "b1", 0.02),
            (0.30, 100.0, 6.0, 0, "b2", 0.03),
        ]
    ):
        pullback = base + pd.Timedelta(minutes=idx * 15)
        rows.append(
            {
                "portfolio_id": P2_PORTFOLIO_ID,
                "trade_id": f"t{idx}",
                "signal_id": f"s{idx}",
                "entry_time": pullback + pd.Timedelta(minutes=reclaim_minutes + 15),
                "pullback_time": pullback,
                "reclaim_time": pullback + pd.Timedelta(minutes=reclaim_minutes),
                "volume_impulse_density_at_signal": density,
                "beta_extension_score_at_signal": beta,
                "local_volume_shock_strength_at_signal": shock,
                "burst_id": burst,
                "burst_count_so_far": idx + 1,
                "uses_final_burst_size_for_decision": False,
                "net_return_20bp": net20,
                "is_overflow": False,
                "timely_forward_observation": True,
            }
        )
    return pd.DataFrame(rows)


def test_fixed_score_maps_frozen_endpoints_without_using_final_burst_size() -> None:
    rows = _rows()
    scored = add_continuation_strength(rows)
    assert scored["continuation_strength"].round(6).tolist() == [0.0, 0.5, 1.0]

    changed = rows.copy()
    changed["burst_count_so_far"] = [100, 200, 300]
    rescored = add_continuation_strength(changed)
    assert rescored["continuation_strength"].tolist() == scored["continuation_strength"].tolist()


def test_forward_population_requires_timely_observation_and_aggregates_by_burst() -> None:
    rows = _rows()
    rows.loc[0, "timely_forward_observation"] = False
    core = prepare_p2_core(rows, forward_only=True)
    assert set(core["trade_id"]) == {"t1", "t2"}
    assert set(core["period"]) == {"timely_forward"}

    bursts = burst_level(core).set_index("burst_id")
    assert bursts.loc["b1", "trades"] == 1
    assert bursts.loc["b1", "burst_net20"] == 0.02 / 8.0
    assert bursts.loc["b2", "burst_score"] == 1.0
