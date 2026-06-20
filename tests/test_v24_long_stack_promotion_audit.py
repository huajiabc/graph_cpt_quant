from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v24_long_stack_promotion_audit import (
    V24Config,
    write_v24_long_stack_promotion_audit,
)


def _write_architecture(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "live_architecture_summary.csv", index=False)


def _base_row(date: str, structure: str, trades: int, net20: float, **kwargs) -> dict:
    row = {
        "date": date,
        "structure": structure,
        "trades": trades,
        "net10": net20 + 0.002,
        "net20": net20,
        "net30": net20 - 0.002,
        "core_pnl": net20,
        "overflow_pnl": 0.0,
        "checkpoint_pnl": 0.0,
        "protect_counterfactual_pnl": 0.0,
        "max_exposure": 8.0,
        "max_concurrent_positions": 8.0,
        "checkpoint_exits": 0,
        "protected_exits": 0,
        "overflow_trades": 0,
    }
    row.update(kwargs)
    return row


def test_v24_promotion_audit_marks_forward_sample_insufficient(tmp_path: Path) -> None:
    v23 = tmp_path / "v23"
    source = tmp_path / "source"
    failure = tmp_path / "failure"
    source.mkdir()
    pd.DataFrame().to_parquet(source / "checkpoint_trade_ledger.parquet", index=False)
    _write_architecture(
        v23,
        [
            _base_row("2026-05-06", "P2_MAX8_BASELINE", 5, -0.008),
            _base_row("2026-05-06", "P2_MAX8_CP60_PLUS_O6", 5, -0.005, checkpoint_pnl=0.003, checkpoint_exits=1),
            _base_row(
                "2026-05-06",
                "P2_MAX8_CP60_PROTECT_A_CAP2_PLUS_O6",
                5,
                -0.005,
                checkpoint_pnl=0.003,
                checkpoint_exits=1,
            ),
        ],
    )
    failure.mkdir()
    pd.DataFrame(
        [
            {
                "structure_id": "R3_P2_MAX8_PROTECT_A_CAP2_O6",
                "risk_off_gated_candidates": 3,
                "risk_off_gated_net20_avg": -0.01,
                "risk_off_false_skip_rate": 0.5,
                "delta_net20_vs_base": -0.01,
                "delta_drawdown_vs_base": 0.01,
            }
        ]
    ).to_csv(failure / "architecture_overlay_summary.csv", index=False)
    pd.DataFrame(
        [
            {"risk_state": "low_coimpulse_high", "net20_later": -0.03},
            {"risk_state": "normal_coimpulse", "net20_later": 0.01},
        ]
    ).to_csv(v23 / "live_regime_diagnostics.csv", index=False)

    outputs = write_v24_long_stack_promotion_audit(
        V24Config(
            report_root=tmp_path / "out",
            v23_root=v23,
            source_root=source,
            failure_overlay_root=failure,
        )
    )

    assert outputs["stack_comparison"].exists()
    suff = pd.read_csv(outputs["forward_sample_sufficiency"])
    assert suff.loc[suff["structure_id"].eq("S5"), "overall_sample_status"].iloc[0] == "insufficient"
    decisions = pd.read_csv(outputs["promotion_decision_table"])
    s5_decision = decisions.loc[decisions["structure_id"].eq("S5"), "decision"].iloc[0]
    assert s5_decision == "KEEP_SHADOW_RESEARCH_IMPROVED_NEED_PROTECT_OR_OVERFLOW_SAMPLE"
    low = pd.read_csv(outputs["low_coimpulse_diagnostic"])
    assert set(low["risk_state"]) == {"low_coimpulse_high", "normal_coimpulse"}


def test_v24_can_promote_sufficient_s5_shadow_candidate(tmp_path: Path) -> None:
    v23 = tmp_path / "v23"
    source = tmp_path / "source"
    failure = tmp_path / "failure"
    source.mkdir()
    pd.DataFrame().to_parquet(source / "checkpoint_trade_ledger.parquet", index=False)
    _write_architecture(
        v23,
        [
            _base_row("2026-05-01", "P2_MAX8_BASELINE", 120, 0.05, max_exposure=8),
            _base_row(
                "2026-05-01",
                "P2_MAX8_CP60_PLUS_O6",
                130,
                0.10,
                max_exposure=9,
                checkpoint_pnl=0.02,
                checkpoint_exits=60,
                overflow_trades=35,
                overflow_pnl=0.01,
            ),
            _base_row(
                "2026-05-01",
                "P2_MAX8_CP60_PROTECT_A_CAP2_PLUS_O6",
                130,
                0.12,
                max_exposure=9,
                checkpoint_pnl=0.02,
                protect_counterfactual_pnl=0.01,
                checkpoint_exits=60,
                protected_exits=35,
                overflow_trades=35,
                overflow_pnl=0.01,
            ),
        ],
    )
    failure.mkdir()
    pd.DataFrame().to_csv(failure / "architecture_overlay_summary.csv", index=False)
    pd.DataFrame().to_csv(v23 / "live_regime_diagnostics.csv", index=False)

    outputs = write_v24_long_stack_promotion_audit(
        V24Config(
            report_root=tmp_path / "out",
            v23_root=v23,
            source_root=source,
            failure_overlay_root=failure,
        )
    )

    decisions = pd.read_csv(outputs["promotion_decision_table"])
    assert (
        decisions.loc[decisions["structure_id"].eq("S5"), "decision"].iloc[0]
        == "PROMOTE_TO_SHADOW_CANDIDATE"
    )
