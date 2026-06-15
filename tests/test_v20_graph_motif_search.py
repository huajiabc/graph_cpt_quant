from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v20_graph_motif_search import (
    MotifSpec,
    V20Config,
    _aco_path_miner,
    _evaluation_row,
    _ga_architecture_search,
    _sa_refinement,
    _simulate_portfolio,
)


def _row(idx: int, *, candidate: str, net: float, checkpoint_net: float, beta: float = 100.0) -> dict[str, object]:
    entry = pd.Timestamp("2025-10-01T00:00:00Z") + pd.Timedelta(minutes=20 * idx)
    return {
        "exchange": "bybit",
        "symbol": f"S{idx}USDT",
        "candidate": candidate,
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=4),
        "checkpoint_time": entry + pd.Timedelta(hours=1),
        "checkpoint_price_covered": True,
        "checkpoint_net_at_cost": checkpoint_net,
        "net_return_at_cost": net,
        "btc_state_at_entry": "BTC_up" if idx % 2 == 0 else "BTC_chop",
        "cluster_density": 0.9 if idx < 3 else 0.1,
        "burst_id": "b0" if idx < 4 else "b1",
        "burst_count_so_far": idx + 1,
        "month": "2025-10",
        "trade_key": f"sig-{idx}",
        "signal_id": f"sig-{idx}",
        "beta_extreme_strength": beta,
        "beta_extreme_strength_high": beta >= 100.0,
    }


def _sample() -> pd.DataFrame:
    rows = [
        _row(0, candidate="CIC1_beta_extreme", net=0.04, checkpoint_net=-0.01),
        _row(1, candidate="CIC2_beta_broad", net=-0.02, checkpoint_net=-0.005, beta=90.0),
        _row(2, candidate="CIC1_beta_extreme", net=0.03, checkpoint_net=-0.006),
        _row(3, candidate="CIC2_beta_broad", net=0.01, checkpoint_net=0.002, beta=90.0),
        _row(4, candidate="CIC1_beta_extreme", net=0.05, checkpoint_net=-0.004),
    ]
    return pd.DataFrame(rows)


def test_v20_simulates_checkpoint_protection_and_overflow() -> None:
    sample = _sample()
    spec = MotifSpec(
        max_positions=1,
        overflow_rule="O6_late9",
        overflow_trigger=1,
        overflow_slots=2,
        cic1_overflow_size=0.5,
        cic2_overflow_size=0.25,
        checkpoint_rule="Protect_A_cap2",
        protect_cap=2,
    )
    selected, skipped = _simulate_portfolio(sample, spec)

    assert not selected.empty
    assert selected["sleeve"].isin(["core", "overflow"]).any()
    protected = selected[selected.get("kept_due_to_protection").fillna(False).astype(bool)]
    assert protected.groupby("burst_id").size().max() <= 2
    assert len(skipped) >= 0


def test_v20_search_components_produce_rows() -> None:
    sample = _sample()
    cfg = V20Config(aco_iterations=2, aco_ants=4, ga_population=8, ga_generations=2, sa_steps=5)

    aco = _aco_path_miner(sample, cfg)
    assert not aco.empty
    assert {"nodes", "fitness"}.issubset(aco.columns)

    seeds = [MotifSpec(), MotifSpec(max_positions=1, checkpoint_rule="CP60")]
    ga = _ga_architecture_search(sample, cfg, seeds)
    assert not ga.empty

    starts = [MotifSpec(max_positions=1, checkpoint_rule="Protect_A_cap2", protect_cap=2)]
    sa = _sa_refinement(sample, cfg, starts)
    assert not sa.empty

    row = _evaluation_row(starts[0], sample, candidate_id="T0", source="test")
    assert row["candidate_id"] == "T0"
    assert "search_portfolio_net20" in row
