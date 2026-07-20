import numpy as np

from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    audit_v2334,
    build_v2334_five_minute_buckets,
    load_v2334_liquidations,
)


def test_v2334_current_forward_snapshot_passes_audit() -> None:
    audit, summary, by_symbol, market = audit_v2334()
    assert audit["passed"].all(), audit.loc[~audit["passed"]].to_dict("records")
    assert len(summary) == 17
    assert int(by_symbol["liquidation_events"].sum()) == int(summary["events"].sum())
    assert int(market["liquidation_events"].sum()) == int(summary["events"].sum())


def test_v2334_five_minute_panels_reconcile_raw_events() -> None:
    events = load_v2334_liquidations()
    by_symbol, market = build_v2334_five_minute_buckets(events)
    assert int(by_symbol["liquidation_events"].sum()) == len(events)
    assert int(market["liquidation_events"].sum()) == len(events)
    assert np.isclose(
        market["total_liquidation_usd"].sum(), events["notional_usd"].sum()
    )
    assert np.isclose(
        market["net_forced_buy_usd"].sum(),
        market["forced_buy_usd"].sum() - market["forced_sell_usd"].sum(),
    )
