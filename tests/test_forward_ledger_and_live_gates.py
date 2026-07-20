from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from pressure_graph.config.v07a2 import load_v07a2_config
from pressure_graph.live.gates import evaluate_live_gates, write_live_gate_artifacts
from pressure_graph.paper_live.forward_ledger import merge_cumulative_ledger, write_forward_run
from pressure_graph.paper_live.v07d2 import (
    REPLAY_DATA_ROOT,
    REPLAY_REPORT_ROOT,
    write_v07d2_cic_mir1_paper_live,
)


def _signal(signal_id: str, feature_time: str, status: str = "detected") -> pd.DataFrame:
    return pd.DataFrame(
        [{"signal_id": signal_id, "feature_time": pd.Timestamp(feature_time), "status": status}]
    )


def _trade_rows(count: int, *, net10: float, baseline: bool = False) -> pd.DataFrame:
    base = pd.Timestamp("2026-07-01T00:00:00Z")
    rows = []
    for idx in range(count):
        row = {
            "trade_id": f"{'b' if baseline else 'p'}-{idx}",
            "signal_id": f"{'bs' if baseline else 'ps'}-{idx}",
            "candidate": "CIC1_FILTERED_MIR1",
            "entry_time": base + pd.Timedelta(minutes=idx),
            "exit_time": base + pd.Timedelta(minutes=idx + 1),
            "exit_reason": "max_hold",
            "net_return_10bp": net10,
            "timely_forward_observation": True,
            "portfolio_accepted": True,
        }
        if baseline:
            row["baseline_kind"] = "matched_random_reclaim"
        rows.append(row)
    return pd.DataFrame(rows)


def test_replay_defaults_are_isolated_from_live_namespace() -> None:
    signature = inspect.signature(write_v07d2_cic_mir1_paper_live)
    assert signature.parameters["report_root"].default == REPLAY_REPORT_ROOT
    assert signature.parameters["paper_data_root"].default == REPLAY_DATA_ROOT
    config = load_v07a2_config("configs/v0_7d2_cic_mir1_paper_live.yaml")
    assert config.forward_primary.portfolio_id == "P2_MAX8_BASELINE"
    assert config.forward_primary.max_positions == 8


def test_cumulative_ledger_preserves_first_observation_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "signals.parquet"
    first_observed = pd.Timestamp("2026-07-10T10:15:00Z")
    first = merge_cumulative_ledger(
        _signal("s1", "2026-07-10T10:00:00Z"),
        path,
        key_cols=["signal_id"],
        observed_at=first_observed,
        event_time_col="feature_time",
        timely_lag_minutes=30,
    )
    assert len(first) == 1
    assert bool(first.iloc[0]["timely_forward_observation"])

    second_observed = pd.Timestamp("2026-07-10T11:00:00Z")
    rolling = pd.concat(
        [
            _signal("s1", "2026-07-10T10:00:00Z", status="exited"),
            _signal("s2", "2026-07-10T08:00:00Z"),
        ],
        ignore_index=True,
    )
    second = merge_cumulative_ledger(
        rolling,
        path,
        key_cols=["signal_id"],
        observed_at=second_observed,
        event_time_col="feature_time",
        timely_lag_minutes=30,
    ).set_index("signal_id")

    assert len(second) == 2
    assert pd.Timestamp(second.loc["s1", "first_observed_at_utc"]) == first_observed
    assert pd.Timestamp(second.loc["s1", "last_observed_at_utc"]) == second_observed
    assert bool(second.loc["s1", "timely_forward_observation"])
    assert not bool(second.loc["s2", "timely_forward_observation"])
    assert second.loc["s1", "status"] == "exited"


def test_forward_run_writes_snapshots_manifest_and_cumulative_ledger(tmp_path: Path) -> None:
    prepared = pd.DataFrame([{"feature_time": pd.Timestamp("2026-07-10T10:00:00Z")}])
    signals = _signal("s1", "2026-07-10T10:00:00Z")
    empty = pd.DataFrame()

    write_forward_run(
        report_root=tmp_path,
        prepared=prepared,
        signals=signals,
        trades=empty,
        baseline_trades=empty,
        portfolio_trades=empty,
        overflow_trades=empty,
        checkpoint_trades=empty,
        data_stale=False,
        observed_at="2026-07-10T10:15:00Z",
    )
    write_forward_run(
        report_root=tmp_path,
        prepared=prepared,
        signals=signals,
        trades=empty,
        baseline_trades=empty,
        portfolio_trades=empty,
        overflow_trades=empty,
        checkpoint_trades=empty,
        data_stale=False,
        observed_at="2026-07-10T10:30:00Z",
    )

    root = tmp_path / "forward"
    assert len(list((root / "runs").iterdir())) == 2
    assert len(pd.read_csv(root / "run_manifest.csv")) == 2
    cumulative = pd.read_parquet(root / "signals.parquet")
    assert len(cumulative) == 1
    assert pd.Timestamp(cumulative.iloc[0]["first_observed_at_utc"]) == pd.Timestamp(
        "2026-07-10T10:15:00Z"
    )


def test_forward_manifest_migrates_legacy_header_and_recovers_wide_row(tmp_path: Path) -> None:
    report_root = tmp_path / "report"
    forward_root = report_root / "forward"
    forward_root.mkdir(parents=True)
    legacy_columns = [
        "run_id",
        "observed_at_utc",
        "latest_feature_time",
        "data_stale",
        "rolling_signals",
        "rolling_trades",
        "rolling_baseline_trades",
        "rolling_portfolio_trades",
        "rolling_overflow_trades",
        "rolling_checkpoint_trades",
        "cumulative_signals",
        "cumulative_trades",
        "cumulative_baseline_trades",
        "cumulative_portfolio_trades",
        "cumulative_overflow_trades",
        "cumulative_checkpoint_trades",
        "timely_signals",
    ]
    current_columns = [
        *legacy_columns[:10],
        "rolling_risk_shadow_trades",
        "rolling_risk_shadow_skipped",
        "rolling_token_context",
        *legacy_columns[10:16],
        "cumulative_risk_shadow_trades",
        "cumulative_risk_shadow_skipped",
        "cumulative_token_context",
        legacy_columns[16],
    ]
    manifest = forward_root / "run_manifest.csv"
    manifest.write_text(
        ",".join(legacy_columns)
        + "\n"
        + ",".join(["legacy", *(["0"] * 16)])
        + "\n"
        + ",".join(["wide", *(["0"] * 18)])
        + "\n",
        encoding="utf-8",
    )
    signals = _signal("new-signal", "2026-07-11T00:00:00Z")
    trades = _trade_rows(1, net10=0.01)
    write_forward_run(
        report_root=report_root,
        prepared=pd.DataFrame({"feature_time": [pd.Timestamp("2026-07-11T00:00:00Z")]}),
        signals=signals,
        trades=trades,
        baseline_trades=trades,
        portfolio_trades=trades.assign(portfolio_id="P2"),
        overflow_trades=trades.assign(portfolio_id="O6"),
        checkpoint_trades=trades.assign(portfolio_id="C60"),
        risk_shadow_trades=trades.assign(risk_shadow_arm="P2_VOL"),
        risk_shadow_skipped=trades.assign(risk_shadow_arm="P2_CORR"),
        token_context=trades,
        data_stale=False,
        observed_at="2026-07-11T00:05:00Z",
    )
    migrated = pd.read_csv(manifest)
    assert list(migrated.columns) == current_columns
    assert list(migrated["run_id"]) == ["legacy", "wide", "20260711T000500000000Z"]
    assert migrated.loc[migrated["run_id"].eq("wide"), "rolling_risk_shadow_trades"].iloc[0] == 0


def test_live_gates_wait_for_sample_then_block_negative_edge() -> None:
    config = load_v07a2_config("configs/v0_7d2_cic_mir1_paper_live.yaml")
    now = pd.Timestamp("2026-07-10T12:00:00Z")
    prepared = pd.DataFrame([{"feature_time": now - pd.Timedelta(minutes=15)}])

    insufficient = evaluate_live_gates(
        prepared,
        _trade_rows(99, net10=-0.01),
        _trade_rows(99, net10=0.0, baseline=True),
        config,
        now=now,
    )
    assert insufficient.allow_new_actions
    assert insufficient.rolling_net_gate_status == "insufficient_sample"

    blocked = evaluate_live_gates(
        prepared,
        _trade_rows(100, net10=-0.01),
        _trade_rows(100, net10=0.0, baseline=True),
        config,
        now=now,
    )
    assert not blocked.allow_new_actions
    assert "rolling_primary_net10_lte_zero" in blocked.reasons
    assert "rolling_baseline_lift10_lte_zero" in blocked.reasons


def test_live_gate_uses_p2_forward_primary_when_portfolio_rows_are_present() -> None:
    config = load_v07a2_config("configs/v0_7d2_cic_mir1_paper_live.yaml")
    now = pd.Timestamp("2026-07-10T12:00:00Z")
    prepared = pd.DataFrame([{"feature_time": now - pd.Timedelta(minutes=15)}])
    p2 = _trade_rows(100, net10=-0.01)
    p2["portfolio_id"] = "P2_MAX8_BASELINE"
    p2["effective_net_return_10bp"] = -0.01
    other = _trade_rows(100, net10=0.05)
    other["portfolio_id"] = "P2_MAX8_CP60"
    other["effective_net_return_10bp"] = 0.05
    decision = evaluate_live_gates(
        prepared,
        pd.concat([p2, other], ignore_index=True),
        _trade_rows(100, net10=-0.02, baseline=True),
        config,
        now=now,
    )
    assert not decision.allow_new_actions
    assert decision.rolling_primary_net10 == pytest.approx(-0.01)
    assert "rolling_primary_net10_lte_zero" in decision.reasons


def test_stale_gate_writes_no_actionable_signals(tmp_path: Path) -> None:
    config = load_v07a2_config("configs/v0_7d2_cic_mir1_paper_live.yaml")
    now = pd.Timestamp("2026-07-10T12:00:00Z")
    prepared = pd.DataFrame([{"feature_time": now - pd.Timedelta(hours=2)}])
    decision = evaluate_live_gates(prepared, pd.DataFrame(), pd.DataFrame(), config, now=now)
    assert not decision.allow_new_actions
    assert decision.data_stale

    signals = _signal("s1", "2026-07-10T11:45:00Z")
    signals["first_observed_at_utc"] = now
    signals["timely_forward_observation"] = True
    outputs = write_live_gate_artifacts(
        report_root=tmp_path,
        decision=decision,
        cumulative_signals=signals,
        observed_at=now,
    )
    assert pd.read_parquet(outputs["actionable_signals"]).empty
