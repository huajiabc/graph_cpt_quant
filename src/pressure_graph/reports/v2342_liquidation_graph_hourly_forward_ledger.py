"""Append-only hourly forward ledger for frozen liquidation graph features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    DATA_ROOT,
    load_v2334_liquidations,
)
from pressure_graph.reports.v2335_liquidation_pre_event_feature_contract import (
    REPORT_ROOT as CONTRACT_ROOT,
    V2335Contract,
    build_v2335_causal_features,
)


LEDGER_PATH = DATA_ROOT / "hourly_graph_feature_ledger.parquet"
REPORT_ROOT = Path("reports/v23_42_liquidation_graph_hourly_forward_ledger")
FINDINGS_PATH = Path(
    "docs/v2342_liquidation_graph_hourly_forward_ledger_2026_07_17.md"
)


def available_v2342_decisions(
    causal_start: pd.Timestamp,
    snapshot_completed_at: pd.Timestamp,
) -> pd.DatetimeIndex:
    start = pd.Timestamp(causal_start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    completed = pd.Timestamp(snapshot_completed_at)
    completed = (
        completed.tz_convert("UTC")
        if completed.tzinfo
        else completed.tz_localize("UTC")
    )
    first = start.ceil("h")
    last = completed.floor("h")
    if first > last:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(first, last, freq="h", tz="UTC")


def update_v2342_ledger(
    data_root: Path = DATA_ROOT,
    ledger_path: Path = LEDGER_PATH,
    contract_root: Path = CONTRACT_ROOT,
) -> tuple[pd.DataFrame, int, dict[str, object]]:
    coverage = json.loads((data_root / "coverage.json").read_text(encoding="utf-8"))
    contract_metadata = json.loads(
        (contract_root / "metadata.json").read_text(encoding="utf-8")
    )
    causal_start = pd.Timestamp(contract_metadata["causal_start"])
    snapshot_completed_at = pd.Timestamp(coverage["batch_completed_at"])
    decisions = available_v2342_decisions(causal_start, snapshot_completed_at)
    existing = pd.read_parquet(ledger_path) if ledger_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing["decision_time"] = pd.to_datetime(
            existing["decision_time"], utc=True, errors="coerce"
        )
    existing_times = (
        set(existing["decision_time"]) if not existing.empty else set()
    )
    missing = pd.DatetimeIndex(
        [decision for decision in decisions if decision not in existing_times]
    )
    if len(missing):
        events = load_v2334_liquidations(data_root)
        appended = build_v2335_causal_features(events, missing)
        appended["source_snapshot_completed_at"] = snapshot_completed_at
        appended["source_total_rows"] = int(coverage["total_rows"])
        appended["knowledge_time_rule"] = "first_seen_at <= decision_time"
        combined = pd.concat([existing, appended], ignore_index=True)
    else:
        combined = existing.copy()
    if not combined.empty:
        combined["decision_time"] = pd.to_datetime(
            combined["decision_time"], utc=True, errors="coerce"
        )
        combined = (
            combined.drop_duplicates("decision_time", keep="first")
            .sort_values("decision_time")
            .reset_index(drop=True)
        )
        ensure_dir(ledger_path.parent)
        temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        combined.to_parquet(temporary, index=False)
        temporary.replace(ledger_path)
    metadata = {
        "status": "forward_hourly_feature_ledger_active",
        "causal_start": causal_start.isoformat(),
        "latest_snapshot_completed_at": snapshot_completed_at.isoformat(),
        "available_hourly_decisions": len(decisions),
        "new_hourly_decisions": len(missing),
        "ledger_rows": len(combined),
        "outcomes_loaded": False,
        "hourly_minimum_decisions": V2335Contract().hourly_minimum_decisions,
        "hourly_minimum_days": V2335Contract().hourly_minimum_days,
    }
    return combined, len(missing), metadata


def audit_v2342(
    ledger: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    if ledger.empty:
        checks = {
            "no_hourly_decision_available_yet": metadata[
                "available_hourly_decisions"
            ]
            == 0,
            "outcomes_not_loaded": metadata["outcomes_loaded"] is False,
        }
    else:
        feature_columns = [
            column
            for column in ledger.columns
            if column.startswith("liq_")
        ]
        checks = {
            "decision_keys_unique": ledger["decision_time"].is_unique,
            "decisions_not_before_frozen_causal_start": ledger[
                "decision_time"
            ].ge(pd.Timestamp(metadata["causal_start"])).all(),
            "decisions_not_after_latest_snapshot": ledger["decision_time"].le(
                pd.Timestamp(metadata["latest_snapshot_completed_at"])
            ).all(),
            "feature_values_finite": np.isfinite(ledger[feature_columns]).all().all(),
            "knowledge_time_rule_frozen": ledger["knowledge_time_rule"].eq(
                "first_seen_at <= decision_time"
            ).all(),
            "no_outcome_columns": not any(
                token in column
                for column in ledger.columns
                for token in ("future", "return", "pnl", "label", "target")
            ),
            "outcomes_not_loaded": metadata["outcomes_loaded"] is False,
        }
    return pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )


def write_v2342(
    data_root: Path = DATA_ROOT,
    ledger_path: Path = LEDGER_PATH,
    contract_root: Path = CONTRACT_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    ledger, new_rows, metadata = update_v2342_ledger(
        data_root, ledger_path, contract_root
    )
    audit = audit_v2342(ledger, metadata)
    root = ensure_dir(report_root)
    paths = {
        "ledger": ledger_path,
        "audit": root / "audit_checks.csv",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    audit.to_csv(paths["audit"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    status = (
        "first_forward_hourly_feature_appended"
        if new_rows > 0 and len(ledger) == new_rows
        else "forward_hourly_features_appended"
        if new_rows > 0
        else "waiting_for_next_complete_hour"
    )
    findings = [
        "# v23.42 Liquidation Graph Hourly Forward Ledger",
        "",
        f"Status: `{status}`.",
        "",
        f"Frozen causal start: {metadata['causal_start']}.",
        f"Latest complete source snapshot: {metadata['latest_snapshot_completed_at']}.",
        f"Ledger rows: {metadata['ledger_rows']}; appended now: {new_rows}.",
        "",
        "Only feature rows are stored. Outcomes remain unloaded until their frozen "
        "1-hour and 4-hour horizons have fully elapsed, and evaluation remains gated "
        "at 336 hourly decisions across at least 14 UTC days.",
        "",
    ]
    ensure_dir(findings_path.parent)
    findings_path.write_text("\n".join(findings), encoding="utf-8")
    return paths


__all__ = [
    "LEDGER_PATH",
    "audit_v2342",
    "available_v2342_decisions",
    "update_v2342_ledger",
    "write_v2342",
]
