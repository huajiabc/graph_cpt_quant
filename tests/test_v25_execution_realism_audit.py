from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v25_execution_realism_audit import (
    V25Config,
    write_v25_execution_realism_audit,
)


def test_v25_execution_realism_reports_cost_and_blockers(tmp_path: Path) -> None:
    v24 = tmp_path / "v24"
    source = tmp_path / "source"
    v24.mkdir()
    source.mkdir()
    pd.DataFrame(
        [
            {
                "structure_id": "S5",
                "structure": "P2_MAX8_CP60_PROTECT_A_CAP2_PLUS_O6",
                "label": "P2 max8 + Protect_A cap2 + O6",
                "trades": 12,
                "net10": 0.03,
                "net20": 0.02,
                "net30": 0.01,
                "net50": None,
                "checkpoint_exits": 7,
                "protected_exits": 0,
                "overflow_trades": 0,
            }
        ]
    ).to_csv(v24 / "stack_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "portfolio_id": "P2_MAX8_CP60",
                "trade_id": "t1",
                "selected": True,
                "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
                "effective_net_return_20bp": 0.01,
                "checkpoint_triggered": True,
                "checkpoint_price": 100.0,
            }
        ]
    ).to_parquet(source / "checkpoint_trade_ledger.parquet", index=False)

    outputs = write_v25_execution_realism_audit(
        V25Config(report_root=tmp_path / "out", v24_root=v24, source_root=source)
    )

    cost = pd.read_csv(outputs["execution_cost_stress"])
    assert float(cost.loc[0, "net50_est"]) == -0.01
    assumptions = pd.read_csv(outputs["execution_assumption_check"])
    assert assumptions["blocking_for_real_live"].astype(str).str.lower().isin(["true"]).any()
    assert outputs["execution_realism_decision"].exists()
