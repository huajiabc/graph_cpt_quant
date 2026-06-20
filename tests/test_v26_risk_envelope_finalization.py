from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v26_risk_envelope_finalization import (
    V26Config,
    write_v26_risk_envelope_finalization,
)


def test_v26_writes_risk_spec_and_not_ready_decision(tmp_path: Path) -> None:
    v24 = tmp_path / "v24"
    v25 = tmp_path / "v25"
    v24.mkdir()
    v25.mkdir()
    pd.DataFrame(
        [
            {
                "structure_id": "S5",
                "structure": "P2_MAX8_CP60_PROTECT_A_CAP2_PLUS_O6",
                "label": "P2 max8 + Protect_A cap2 + O6",
                "max_exposure": 9.0,
            }
        ]
    ).to_csv(v24 / "stack_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "structure_id": "S5",
                "overall_sample_status": "insufficient",
            }
        ]
    ).to_csv(v24 / "forward_sample_sufficiency.csv", index=False)
    pd.DataFrame(
        [
            {
                "structure_id": "S5",
                "risk_envelope_status": "not_passed",
            }
        ]
    ).to_csv(v24 / "risk_envelope_check.csv", index=False)
    pd.DataFrame(
        [
            {
                "structure_id": "S5",
                "cost_stress_status": "not_passed",
            }
        ]
    ).to_csv(v25 / "execution_cost_stress.csv", index=False)
    pd.DataFrame(
        [
            {
                "check_id": "depth_slippage",
                "blocking_for_real_live": True,
            }
        ]
    ).to_csv(v25 / "execution_assumption_check.csv", index=False)
    pd.DataFrame(
        [
            {
                "structure_id": "S5",
                "decision": "KEEP_SHADOW_RESEARCH_IMPROVED_NEED_PROTECT_OR_OVERFLOW_SAMPLE",
            }
        ]
    ).to_csv(v24 / "promotion_decision_table.csv", index=False)

    outputs = write_v26_risk_envelope_finalization(
        V26Config(report_root=tmp_path / "out", v24_root=v24, v25_root=v25)
    )

    spec = pd.read_csv(outputs["risk_policy_spec"])
    assert "total_exposure_cap" in set(spec["policy_key"])
    checks = pd.read_csv(outputs["current_envelope_check"])
    assert not checks["passed"].all()
    assert "canary_or_real_live_ready: false" in outputs["risk_envelope_decision"].read_text(
        encoding="utf-8"
    )
