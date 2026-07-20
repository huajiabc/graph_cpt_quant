"""Audit the v23.6 two-sigma OCO temporal confirmation result."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    V234Config,
    build_v234_month_bootstrap,
    build_v234_random_paths,
    load_v234_inputs,
    simulate_v234_oco,
    summarize_v234,
)
from pressure_graph.reports.v236_two_sigma_oco_temporal_confirmation import (
    V236Config,
    decide_v236,
)


V234_ROOT = Path("reports/v23_4_book_vacuum_oco_breakout")
V236_ROOT = Path("reports/v23_6_two_sigma_oco_temporal_confirmation")
REPORT_ROOT = Path("reports/v23_7_two_sigma_oco_temporal_confirmation_audit")
FINDINGS_PATH = Path(
    "docs/v237_two_sigma_oco_temporal_confirmation_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V237Config:
    v234_root: Path = V234_ROOT
    v236_root: Path = V236_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    sigma_multiple: float = 2.0
    tolerance: float = 1e-12


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column.endswith("_time"):
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def _max_errors(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "upper_stop_price",
        "lower_stop_price",
        "exit_spot",
        "trigger_delay_minutes",
        "fill_price",
        "gross_return",
        "primary_net_return",
        "stress_net_return",
        "reversed_primary_net_return",
    ]
    merged = left.merge(right, on="entry_time", suffixes=("_audit", "_saved"))
    rows = []
    for field in fields:
        a = merged[f"{field}_audit"].to_numpy(dtype=float)
        b = merged[f"{field}_saved"].to_numpy(dtype=float)
        error = np.abs(a - b)
        error = error[np.isfinite(error)]
        rows.append(
            {
                "field": field,
                "maximum_absolute_error": float(error.max() if len(error) else 0.0),
            }
        )
    return pd.DataFrame(rows)


def run_v237_audit(
    cfg: V237Config = V237Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, _, bars = load_v234_inputs(V234Config())
    universe = _utc(pd.read_parquet(cfg.v234_root / "causal_control_universe.parquet"))
    pools = _utc(pd.read_parquet(cfg.v234_root / "matched_control_pools.parquet"))
    saved = _utc(pd.read_parquet(cfg.v236_root / "two_sigma_event_outcomes.parquet"))
    saved_controls = _utc(
        pd.read_parquet(cfg.v236_root / "two_sigma_control_outcomes.parquet")
    )
    saved_full_random = pd.read_parquet(
        cfg.v236_root / "full_matched_random_paths.parquet"
    )
    saved_holdout_random = pd.read_parquet(
        cfg.v236_root / "holdout_matched_random_paths.parquet"
    )
    saved_bootstrap = pd.read_parquet(cfg.v236_root / "month_block_bootstrap.parquet")
    saved_summary = pd.read_csv(cfg.v236_root / "result_summary.csv")
    saved_decision = pd.read_csv(cfg.v236_root / "decision_gates.csv")
    sim_cfg = V234Config()
    audit = simulate_v234_oco(
        features, bars, sim_cfg, sigma_multiple=cfg.sigma_multiple
    )
    audit_controls = simulate_v234_oco(
        universe, bars, sim_cfg, sigma_multiple=cfg.sigma_multiple
    )
    errors = _max_errors(audit, saved)
    control_errors = _max_errors(audit_controls, saved_controls)
    control_errors["field"] = "control_" + control_errors["field"]
    errors = pd.concat([errors, control_errors], ignore_index=True)

    full_random = build_v234_random_paths(audit, audit_controls, pools, sim_cfg)
    holdout_times = set(audit.loc[audit["period"].eq("holdout"), "entry_time"])
    holdout = audit[audit["entry_time"].isin(holdout_times)]
    holdout_pools = pools[pools["event_time"].isin(holdout_times)]
    holdout_random = build_v234_random_paths(
        holdout, audit_controls, holdout_pools, sim_cfg
    )
    bootstrap = build_v234_month_bootstrap(audit, sim_cfg)
    summary = summarize_v234(audit)
    summary["candidate"] = saved_summary["candidate"].iloc[0]
    decision, verdict = decide_v236(
        summary,
        full_random,
        holdout_random,
        bootstrap,
        V236Config(),
    )
    numeric_random = [
        "event_mean_primary_net",
        "control_mean_primary_net",
        "event_minus_control",
    ]
    full_random_error = float(
        np.max(
            np.abs(
                full_random[numeric_random].to_numpy()
                - saved_full_random[numeric_random].to_numpy()
            )
        )
    )
    holdout_random_error = float(
        np.max(
            np.abs(
                holdout_random[numeric_random].to_numpy()
                - saved_holdout_random[numeric_random].to_numpy()
            )
        )
    )
    bootstrap_error = float(
        np.max(
            np.abs(
                bootstrap["mean_primary_net_return"].to_numpy()
                - saved_bootstrap["mean_primary_net_return"].to_numpy()
            )
        )
    )
    summary_fields = [
        "triggered_trades",
        "mean_primary_net_return_per_event_bp",
        "mean_stress_net_return_per_event_bp",
        "mean_reversed_primary_net_return_per_event_bp",
    ]
    summary_merge = summary.merge(saved_summary, on="scope", suffixes=("_audit", "_saved"))
    summary_error = max(
        float(
            np.max(
                np.abs(
                    summary_merge[f"{field}_audit"].to_numpy(dtype=float)
                    - summary_merge[f"{field}_saved"].to_numpy(dtype=float)
                )
            )
        )
        for field in summary_fields
    )
    decision_error = float(
        np.max(
            np.abs(
                decision["observed"].to_numpy()
                - saved_decision["observed"].to_numpy()
            )
        )
    )
    checks = {
        "all_event_and_control_paths_replayed_exactly": float(
            errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "two_sigma_width_exact": bool(
            np.allclose(
                np.log(saved["upper_stop_price"] / saved["entry_spot"]),
                cfg.sigma_multiple * saved["causal_hourly_sigma"],
                atol=cfg.tolerance,
            )
        ),
        "all_159_event_keys_preserved": set(saved["entry_time"])
        == set(features["entry_time"]),
        "holdout_has_49_events_and_25_triggers": len(holdout) == 49
        and int(holdout["triggered"].sum()) == 25,
        "full_random_paths_exact": full_random_error <= cfg.tolerance,
        "holdout_random_paths_exact": holdout_random_error <= cfg.tolerance,
        "month_bootstrap_exact": bootstrap_error <= cfg.tolerance,
        "summary_exact": summary_error <= cfg.tolerance,
        "decision_exact": decision_error <= cfg.tolerance
        and decision["passed"].eq(saved_decision["passed"]).all(),
        "holdout_confirmation_failed": not bool(
            saved_decision.loc[
                saved_decision["gate"].eq("holdout_primary_net_positive"), "passed"
            ].iloc[0]
        ),
        "full_relative_edge_preserved": bool(
            saved_decision.loc[
                saved_decision["gate"].eq(
                    "full_matched_random_percentile_at_least_90"
                ),
                "passed",
            ].iloc[0]
        ),
        "verdict_recomputed_as_rejection": verdict == "two_sigma_oco_rejected",
    }
    checks_frame = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "full_random_maximum_error", "value": full_random_error},
            {
                "diagnostic": "holdout_random_maximum_error",
                "value": holdout_random_error,
            },
            {"diagnostic": "bootstrap_maximum_error", "value": bootstrap_error},
            {"diagnostic": "summary_maximum_error", "value": summary_error},
            {"diagnostic": "decision_maximum_error", "value": decision_error},
        ]
    )
    return checks_frame, errors, diagnostics


def write_v237_two_sigma_oco_temporal_confirmation_audit(
    cfg: V237Config = V237Config(),
) -> dict[str, Path]:
    checks, errors, diagnostics = run_v237_audit(cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "checks": root / "audit_checks.csv",
        "errors": root / "maximum_errors.csv",
        "diagnostics": root / "audit_diagnostics.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    checks.to_csv(paths["checks"], index=False)
    errors.to_csv(paths["errors"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    passed = bool(checks["passed"].all())
    paths["metadata"].write_text(
        json.dumps(
            {
                "audit_passed": passed,
                "checks_passed": int(checks["passed"].sum()),
                "checks_total": len(checks),
                "validated_verdict": "two_sigma_oco_rejected" if passed else "audit_failed",
                "permissions_changed": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "audit_pass_validates_rejection" if passed else "audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.7 Two-Sigma OCO Temporal Confirmation Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "Two-sigma event/control paths, full and holdout matched random",
                "paths, month bootstrap, summary, and rejection were replayed.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V237Config",
    "run_v237_audit",
    "write_v237_two_sigma_oco_temporal_confirmation_audit",
]
