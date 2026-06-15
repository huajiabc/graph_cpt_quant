from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v20_graph_motif_search import MotifSpec
from pressure_graph.reports.v21a_holdout_autopsy import (
    _group_summary,
    _ledger_for_spec,
    _selected_vs_skipped_summary,
)


def _row(idx: int, *, candidate: str, net: float, checkpoint_net: float, beta: float = 100.0) -> dict[str, object]:
    entry = pd.Timestamp("2026-05-10T00:00:00Z") + pd.Timedelta(minutes=15 * idx)
    return {
        "symbol": f"S{idx}USDT",
        "candidate": candidate,
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=3),
        "checkpoint_time": entry + pd.Timedelta(hours=1),
        "checkpoint_price_covered": True,
        "checkpoint_net_at_cost": checkpoint_net,
        "net_return_at_cost": net,
        "btc_state_at_entry": "BTC_chop" if idx % 2 else "BTC_up",
        "volume_impulse_density": 0.2 + idx * 0.1,
        "cluster_impulse_density": 0.1 + idx * 0.05,
        "beta_extreme_strength": beta,
        "beta_extreme_strength_high": beta >= 100.0,
        "burst_count_so_far": idx + 1,
        "burst_id": "burst_a",
        "month": "2026-05",
        "trade_key": f"sig-{idx}",
        "signal_id": f"sig-{idx}",
    }


def test_holdout_autopsy_ledger_marks_components_and_groups() -> None:
    sample = pd.DataFrame(
        [
            _row(0, candidate="CIC1_beta_extreme", net=0.05, checkpoint_net=-0.01, beta=101.0),
            _row(1, candidate="CIC2_beta_broad", net=-0.04, checkpoint_net=-0.02, beta=80.0),
            _row(2, candidate="CIC1_beta_extreme", net=0.03, checkpoint_net=-0.005, beta=101.0),
            _row(9, candidate="CIC2_beta_broad", net=0.02, checkpoint_net=0.001, beta=80.0),
        ]
    )
    spec = MotifSpec(
        max_positions=1,
        overflow_rule="O6_late9",
        overflow_trigger=9,
        overflow_slots=2,
        cic1_overflow_size=0.5,
        cic2_overflow_size=0.25,
        checkpoint_rule="Protect_A_cap2",
        protect_cap=2,
    )

    ledger, skipped = _ledger_for_spec(sample, "T0", spec)

    assert not ledger.empty
    assert {"checkpoint_component", "state_cluster", "portfolio_contribution_net20"}.issubset(ledger.columns)
    assert ledger["checkpoint_component"].isin(["ProtectA_kept", "CP60_exit", "O6_overflow", "core_normal"]).any()

    by_phase = _group_summary(ledger, ["candidate_id", "burst_phase"])
    assert not by_phase.empty
    assert by_phase["selected_trades"].sum() == len(ledger)

    selected_vs_skipped = _selected_vs_skipped_summary(ledger, skipped)
    assert selected_vs_skipped.iloc[0]["candidate_id"] == "T0"
    assert selected_vs_skipped.iloc[0]["selected_trades"] == len(ledger)
