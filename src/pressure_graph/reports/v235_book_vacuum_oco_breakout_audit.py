"""Independent audit of the v23.4 BTC OCO breakout result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    V233Config,
    load_v233_btc_15m,
)
from pressure_graph.reports.v234_book_vacuum_oco_breakout import FEATURE_SHA256


V233_ROOT = Path("reports/v23_3_book_vacuum_oco_breakout_feature_audit")
V234_ROOT = Path("reports/v23_4_book_vacuum_oco_breakout")
RAW_EVENT_PATH = Path(
    "reports/v22_4_alt_book_vacuum_pressure_feature_audit/"
    "candidate_feature_events.parquet"
)
REPORT_ROOT = Path("reports/v23_5_book_vacuum_oco_breakout_audit")
FINDINGS_PATH = Path("docs/v235_book_vacuum_oco_breakout_audit_2026_07_17.md")


@dataclass(frozen=True)
class V235Config:
    v233_root: Path = V233_ROOT
    v234_root: Path = V234_ROOT
    raw_event_path: Path = RAW_EVENT_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    path_hours: int = 4
    bar_minutes: int = 15
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    event_exclusion_hours: int = 8
    nearest_controls: int = 10
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    seed: int = 20260717
    tolerance: float = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column.endswith("_time"):
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def _manual_outcomes(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V235Config,
) -> pd.DataFrame:
    indexed = bars.set_index("bar_open_time").sort_index()
    count = cfg.path_hours * 60 // cfg.bar_minutes
    rows = []
    for event in features.itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        times = [
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset)
            for offset in range(count)
        ]
        path = indexed.loc[times]
        upper = float(event.upper_stop_price)
        lower = float(event.lower_stop_price)
        exit_spot = float(path.iloc[-1]["close"])
        triggered = False
        ambiguous = False
        direction = 0
        trigger_time = pd.NaT
        fill = np.nan
        gross = 0.0
        reverse = 0.0
        for time, bar in path.iterrows():
            up = float(bar["high"]) >= upper
            down = float(bar["low"]) <= lower
            if not up and not down:
                continue
            triggered = True
            trigger_time = pd.Timestamp(time)
            long_fill = max(upper, float(bar["open"]))
            short_fill = min(lower, float(bar["open"]))
            long_return = exit_spot / long_fill - 1.0
            short_return = 1.0 - exit_spot / short_fill
            if up and down:
                ambiguous = True
                if long_return <= short_return:
                    direction, fill, gross = 1, long_fill, long_return
                else:
                    direction, fill, gross = -1, short_fill, short_return
                reverse = min(
                    1.0 - exit_spot / long_fill,
                    exit_spot / short_fill - 1.0,
                )
            elif up:
                direction, fill, gross = 1, long_fill, long_return
                reverse = 1.0 - exit_spot / long_fill
            else:
                direction, fill, gross = -1, short_fill, short_return
                reverse = exit_spot / short_fill - 1.0
            break
        cost = cfg.primary_cost if triggered else 0.0
        stress = cfg.stress_cost if triggered else 0.0
        rows.append(
            {
                "entry_time": entry,
                "exit_spot": exit_spot,
                "triggered": triggered,
                "ambiguous_trigger": ambiguous,
                "trigger_time": trigger_time,
                "trigger_delay_minutes": (
                    (trigger_time - entry).total_seconds() / 60.0
                    if triggered
                    else np.nan
                ),
                "trade_direction": direction,
                "fill_price": fill,
                "gross_return": gross,
                "primary_net_return": gross - cost,
                "stress_net_return": gross - stress,
                "reversed_primary_net_return": reverse - cost,
            }
        )
    return pd.DataFrame(rows)


def _outcome_errors(audit: pd.DataFrame, saved: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "exit_spot",
        "trigger_delay_minutes",
        "fill_price",
        "gross_return",
        "primary_net_return",
        "stress_net_return",
        "reversed_primary_net_return",
    ]
    merged = audit.merge(saved, on="entry_time", suffixes=("_audit", "_saved"))
    rows = []
    for field in fields:
        left = merged[f"{field}_audit"].to_numpy(dtype=float)
        right = merged[f"{field}_saved"].to_numpy(dtype=float)
        error = np.abs(left - right)
        error = error[np.isfinite(error)]
        rows.append(
            {
                "field": field,
                "maximum_absolute_error": float(error.max() if len(error) else 0.0),
            }
        )
    for field in ("triggered", "ambiguous_trigger", "trade_direction"):
        rows.append(
            {
                "field": field,
                "maximum_absolute_error": float(
                    (~merged[f"{field}_audit"].eq(merged[f"{field}_saved"])).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _expected_pairs(
    features: pd.DataFrame,
    universe: pd.DataFrame,
    nearest: int,
) -> set[tuple[pd.Timestamp, pd.Timestamp]]:
    pairs = set()
    for event in features.itertuples(index=False):
        local = universe[
            universe["entry_month"].eq(event.entry_month)
            & universe["utc_hour"].eq(event.entry_time.hour)
        ].copy()
        local["distance"] = np.log(
            local["causal_hourly_sigma"] / event.causal_hourly_sigma
        ).abs()
        local = local.sort_values(["distance", "entry_time"]).head(nearest)
        pairs.update((event.entry_time, time) for time in local["entry_time"])
    return pairs


def _replay_random(
    outcomes: pd.DataFrame,
    controls: pd.DataFrame,
    pools: pd.DataFrame,
    cfg: V235Config,
) -> pd.DataFrame:
    event_lookup = outcomes.set_index("entry_time")["primary_net_return"]
    control_lookup = controls.set_index("entry_time")["primary_net_return"]
    grouped = {
        event: local["control_time"].tolist()
        for event, local in pools.groupby("event_time", sort=True)
    }
    event_mean = float(event_lookup.loc[sorted(grouped)].mean())
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for iteration in range(cfg.random_iterations):
        sampled = [
            values[int(rng.integers(0, len(values)))]
            for values in grouped.values()
        ]
        control_mean = float(control_lookup.loc[sampled].mean())
        rows.append((iteration, event_mean, control_mean, event_mean - control_mean))
    return pd.DataFrame(
        rows,
        columns=[
            "iteration",
            "event_mean_primary_net",
            "control_mean_primary_net",
            "event_minus_control",
        ],
    )


def _replay_bootstrap(outcomes: pd.DataFrame, cfg: V235Config) -> pd.DataFrame:
    groups = {
        month: local["primary_net_return"].to_numpy(dtype=float)
        for month, local in outcomes.groupby("entry_month", sort=True)
    }
    months = sorted(groups)
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
        sampled = rng.choice(months, size=len(months), replace=True)
        rows.append(
            (
                iteration,
                float(np.concatenate([groups[month] for month in sampled]).mean()),
            )
        )
    return pd.DataFrame(rows, columns=["iteration", "mean_primary_net_return"])


def run_v235_audit(
    cfg: V235Config = V235Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _utc(pd.read_parquet(cfg.v233_root / "oco_breakout_features.parquet"))
    outcomes = _utc(pd.read_parquet(cfg.v234_root / "oco_event_outcomes.parquet"))
    universe = _utc(pd.read_parquet(cfg.v234_root / "causal_control_universe.parquet"))
    pools = _utc(pd.read_parquet(cfg.v234_root / "matched_control_pools.parquet"))
    controls = _utc(pd.read_parquet(cfg.v234_root / "oco_control_outcomes.parquet"))
    random_saved = pd.read_parquet(cfg.v234_root / "matched_random_paths.parquet")
    bootstrap_saved = pd.read_parquet(cfg.v234_root / "month_block_bootstrap.parquet")
    decision = pd.read_csv(cfg.v234_root / "decision_gates.csv")
    bars = load_v233_btc_15m(V233Config())
    audit = _manual_outcomes(features, bars, cfg)
    errors = _outcome_errors(audit, outcomes)
    expected_pairs = _expected_pairs(features, universe, cfg.nearest_controls)
    saved_pairs = set(zip(pools["event_time"], pools["control_time"], strict=True))
    raw_events = _utc(pd.read_parquet(cfg.raw_event_path))
    min_distance = min(
        abs((control - event).total_seconds()) / 3600.0
        for control in universe["entry_time"]
        for event in raw_events["entry_time"]
        if abs((control - event).total_seconds()) <= 24 * 3600
    )
    random_audit = _replay_random(outcomes, controls, pools, cfg)
    bootstrap_audit = _replay_bootstrap(outcomes, cfg)
    random_error = float(
        np.max(
            np.abs(
                random_audit.iloc[:, 1:].to_numpy()
                - random_saved.iloc[:, 2:].to_numpy()
            )
        )
    )
    bootstrap_error = float(
        np.max(
            np.abs(
                bootstrap_audit["mean_primary_net_return"].to_numpy()
                - bootstrap_saved["mean_primary_net_return"].to_numpy()
            )
        )
    )
    checks = {
        "v233_feature_audit_passed": bool(
            pd.read_csv(cfg.v233_root / "data_quality_checks.csv")["passed"].all()
        ),
        "feature_hash_matches_preregistration": _sha256(
            cfg.v233_root / "oco_breakout_features.parquet"
        )
        == FEATURE_SHA256,
        "all_159_event_keys_match": set(outcomes["entry_time"])
        == set(features["entry_time"]),
        "trigger_fill_and_return_paths_recomputed_exactly": float(
            errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "only_two_same_bar_ambiguities": int(outcomes["ambiguous_trigger"].sum()) == 2,
        "primary_and_stress_costs_exact": bool(
            np.allclose(
                outcomes["primary_net_return"],
                outcomes["gross_return"]
                - outcomes["triggered"].astype(float) * cfg.primary_cost,
                atol=cfg.tolerance,
            )
            and np.allclose(
                outcomes["stress_net_return"],
                outcomes["gross_return"]
                - outcomes["triggered"].astype(float) * cfg.stress_cost,
                atol=cfg.tolerance,
            )
        ),
        "matched_controls_share_month_and_hour": bool(
            pools["event_month"].eq(pools["control_time"].dt.strftime("%Y-%m")).all()
            and pools["event_time"].dt.hour.eq(pools["control_time"].dt.hour).all()
        ),
        "controls_are_more_than_8h_from_events": min_distance
        > cfg.event_exclusion_hours,
        "nearest_control_pairs_exact": saved_pairs == expected_pairs,
        "all_events_have_5_to_10_controls": pools["event_time"].nunique() == 159
        and pools.groupby("event_time").size().between(5, 10).all(),
        "all_1000_random_paths_replayed_exactly": len(random_saved)
        == cfg.random_iterations
        and random_error <= cfg.tolerance,
        "all_5000_month_bootstraps_replayed_exactly": len(bootstrap_saved)
        == cfg.bootstrap_iterations
        and bootstrap_error <= cfg.tolerance,
        "failed_absolute_and_random_gates_force_rejection": bool(
            not decision.loc[
                decision["gate"].eq("overall_primary_net_positive"), "passed"
            ].iloc[0]
            and not decision.loc[
                decision["gate"].eq("matched_random_percentile_at_least_90"),
                "passed",
            ].iloc[0]
        ),
        "findings_record_rejection": (
            "Verdict: `oco_breakout_rejected`."
            in (
                cfg.v234_root.parent.parent
                / "docs/v234_book_vacuum_oco_breakout_findings_2026_07_17.md"
            ).read_text(encoding="utf-8")
        ),
    }
    checks_frame = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "minimum_control_event_distance_hours", "value": min_distance},
            {"diagnostic": "random_maximum_error", "value": random_error},
            {"diagnostic": "bootstrap_maximum_error", "value": bootstrap_error},
        ]
    )
    return checks_frame, errors, diagnostics


def write_v235_book_vacuum_oco_breakout_audit(
    cfg: V235Config = V235Config(),
) -> dict[str, Path]:
    checks, errors, diagnostics = run_v235_audit(cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "checks": root / "independent_audit_checks.csv",
        "errors": root / "maximum_path_errors.csv",
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
                "validated_verdict": "oco_breakout_rejected" if passed else "audit_failed",
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
                "# v23.5 BTC OCO Breakout Independent Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "All 159 trigger paths, pessimistic ambiguity fills, costs, control",
                "matching, 1,000 random paths, 5,000 bootstraps, and the rejection",
                "were independently replayed.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V235Config",
    "run_v235_audit",
    "write_v235_book_vacuum_oco_breakout_audit",
]
