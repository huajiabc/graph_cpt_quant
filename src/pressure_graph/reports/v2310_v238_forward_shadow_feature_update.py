"""Isolated first forward feature update for the frozen v23.8 candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v161_hourly_liquidity_withdrawal_amplification import (
    load_v161_features,
)
from pressure_graph.reports.v224_alt_book_vacuum_pressure_feature_audit import (
    FEATURE_ROOT,
    V224Config,
    add_v224_symbol_states,
    build_v224_bucket_states,
    select_v224_events,
)


FORWARD_ROOT = Path("data/external/v238_forward_shadow")
FORWARD_FEATURE_ROOT = FORWARD_ROOT / "book_depth/hourly_features"
COLLECTION_MANIFEST = FORWARD_ROOT / "forward_collection_manifest.json"
REPORT_ROOT = Path("reports/v23_10_v238_forward_shadow_feature_update")
FINDINGS_PATH = Path("docs/v2310_v238_forward_shadow_feature_update_2026_07_17.md")
CANDIDATE = "DVB5_POSITIVE_PRESSURE_0625SIGMA_BTC_BREAKOUT"


@dataclass(frozen=True)
class V2310Config:
    historical_feature_root: Path = FEATURE_ROOT
    forward_feature_root: Path = FORWARD_FEATURE_ROOT
    collection_manifest: Path = COLLECTION_MANIFEST
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH


def run_v2310_feature_update(
    cfg: V2310Config = V2310Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    historical = load_v161_features(cfg.historical_feature_root)
    forward = load_v161_features(cfg.forward_feature_root)
    cutoff = pd.Timestamp(historical["decision_time"].max())
    combined = (
        pd.concat([historical, forward], ignore_index=True)
        .drop_duplicates(["symbol", "decision_time"], keep="last")
        .sort_values(["symbol", "decision_time"])
        .reset_index(drop=True)
    )
    v224_cfg = V224Config()
    symbol_states = add_v224_symbol_states(combined, v224_cfg)
    bucket_states = build_v224_bucket_states(symbol_states, v224_cfg)
    events = select_v224_events(bucket_states, v224_cfg)
    forward_events = events[events["entry_time"].gt(cutoff)].copy()
    forward_events["forward_candidate_eligible"] = forward_events[
        "signal_direction"
    ].eq(1)
    manifest = json.loads(cfg.collection_manifest.read_text(encoding="utf-8"))
    forward_hours = int(forward["decision_time"].nunique())
    summary = {
        "candidate": CANDIDATE,
        "historical_cutoff": cutoff.isoformat(),
        "forward_first_decision": pd.Timestamp(forward["decision_time"].min()).isoformat(),
        "forward_last_decision": pd.Timestamp(forward["decision_time"].max()).isoformat(),
        "forward_symbol_hours": len(forward),
        "forward_hours": forward_hours,
        "forward_days": forward_hours / 24,
        "forward_bucket_hours": int(
            bucket_states.loc[
                bucket_states["decision_time"].gt(cutoff), "decision_time"
            ].nunique()
        ),
        "strict_v224_forward_events": len(forward_events),
        "positive_pressure_forward_events": int(
            forward_events["forward_candidate_eligible"].sum()
        ),
        "outcomes_loaded": False,
    }
    return forward, bucket_states, forward_events, {**manifest, **summary}


def audit_v2310_feature_update(
    forward: pd.DataFrame,
    bucket_states: pd.DataFrame,
    forward_events: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    cutoff = pd.Timestamp(metadata["historical_cutoff"])
    first = pd.Timestamp(metadata["forward_first_decision"])
    expected_symbols = 16
    ready_symbols = int(
        metadata.get("book_symbols_ready", metadata["book_symbols_downloaded"])
    )
    per_symbol = forward.groupby("symbol")["decision_time"].nunique()
    expected_hours = int(metadata.get("forward_hours", per_symbol.max()))
    expected_times = pd.date_range(
        first, periods=expected_hours, freq="h", tz="UTC"
    )
    actual_times = pd.DatetimeIndex(
        pd.to_datetime(
            forward["decision_time"], utc=True, errors="coerce"
        ).dropna().unique()
    ).sort_values()
    checks = {
        "collection_ready_for_all_16_symbols": ready_symbols == expected_symbols,
        "collection_has_no_missing_book_symbols": metadata["book_symbols_missing"] == 0,
        "hourly_features_ready_for_all_16_symbols": metadata["hourly_symbols_ready"]
        == expected_symbols,
        "forward_panel_has_16_symbols": forward["symbol"].nunique()
        == expected_symbols,
        "each_symbol_has_all_forward_hours": bool(
            per_symbol.eq(expected_hours).all()
        ),
        "forward_hours_are_contiguous": actual_times.equals(expected_times),
        "forward_starts_one_hour_after_cutoff": first
        == cutoff + pd.Timedelta(hours=1),
        "forward_keys_unique": not forward.duplicated(
            ["symbol", "decision_time"]
        ).any(),
        "forward_bucket_has_all_complete_hours": int(
            metadata["forward_bucket_hours"]
        )
        == expected_hours,
        "strict_events_use_frozen_q90_rule": bool(
            forward_events.empty
            or forward_events["bucket_pressure"]
            .abs()
            .ge(forward_events["prior_abs_pressure_threshold"])
            .all()
        ),
        "candidate_requires_positive_pressure": bool(
            forward_events.loc[
                forward_events["forward_candidate_eligible"], "signal_direction"
            ].eq(1).all()
        ),
        "no_forward_outcomes_loaded": metadata["outcomes_loaded"] is False,
        "bucket_states_extend_past_cutoff": bucket_states["decision_time"].max()
        > cutoff,
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def write_v2310_v238_forward_shadow_feature_update(
    cfg: V2310Config = V2310Config(),
) -> dict[str, Path]:
    forward, bucket_states, events, metadata = run_v2310_feature_update(cfg)
    checks = audit_v2310_feature_update(forward, bucket_states, events, metadata)
    root = ensure_dir(cfg.report_root)
    paths = {
        "forward_features": root / "isolated_forward_symbol_features.parquet",
        "bucket_states": root / "forward_bucket_states.parquet",
        "events": root / "new_forward_events.parquet",
        "checks": root / "data_quality_checks.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    forward.to_parquet(paths["forward_features"], index=False)
    cutoff = pd.Timestamp(metadata["historical_cutoff"])
    bucket_states[bucket_states["decision_time"].gt(cutoff)].to_parquet(
        paths["bucket_states"], index=False
    )
    events.to_parquet(paths["events"], index=False)
    checks.to_csv(paths["checks"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    status = (
        "new_frozen_candidate_event_ready"
        if int(metadata["positive_pressure_forward_events"]) > 0
        else "forward_data_ready_no_frozen_event"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.10 v23.8 Forward-Shadow Feature Update",
                "",
                f"Status: `{status}`.",
                "",
                f"Historical cutoff: {metadata['historical_cutoff']}.",
                f"Forward window: {metadata['forward_first_decision']} through",
                f"{metadata['forward_last_decision']}.",
                f"Strict v22.4 events: {metadata['strict_v224_forward_events']}.",
                "Positive-pressure candidate events: "
                f"{metadata['positive_pressure_forward_events']}.",
                "",
                "All 16 book-depth archives and 24 hourly states were collected",
                "under an isolated forward root. No BTC high/low or return outcome",
                "was loaded because the frozen candidate did not fire.",
                "",
                "No parameter, PaperLive, live, leverage, remote, application, or",
                "order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2310Config",
    "audit_v2310_feature_update",
    "run_v2310_feature_update",
    "write_v2310_v238_forward_shadow_feature_update",
]
