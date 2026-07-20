"""Independent audit of the v22.5 alt-book pressure propagation result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
    load_v155_hourly_prices,
)


V224_ROOT = Path("reports/v22_4_alt_book_vacuum_pressure_feature_audit")
V225_ROOT = Path("reports/v22_5_alt_book_vacuum_pressure_to_btc")
REPORT_ROOT = Path("reports/v22_6_alt_book_vacuum_pressure_audit")
FINDINGS_PATH = Path(
    "docs/v226_alt_book_vacuum_pressure_audit_2026_07_17.md"
)
FEATURE_SHA256 = "A6495D01FD26E05D1762531590A886176CCFDF3AFAE870559072A0132C07D43F"


@dataclass(frozen=True)
class V226Config:
    v224_root: Path = V224_ROOT
    v225_root: Path = V225_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    btc_primary_cost: float = 0.0010
    btc_stress_cost: float = 0.0020
    alt_primary_cost: float = 0.0020
    alt_stress_cost: float = 0.0030
    random_iterations: int = 1000
    bootstrap_iterations: int = 2000
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
    for column in (
        "feature_time",
        "entry_time",
        "exit_time",
        "entry_day",
        "decision_time",
    ):
        if column in output:
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def _matrix() -> pd.DataFrame:
    prices = load_v155_hourly_prices()
    prices["feature_time"] = pd.to_datetime(
        prices["feature_time"], utc=True, errors="coerce"
    )
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()


def _reprice(
    features: pd.DataFrame,
    matrix: pd.DataFrame,
    cfg: V226Config,
) -> pd.DataFrame:
    rows = []
    for event in features.sort_values("entry_time").itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        times = [entry + pd.Timedelta(hours=offset) for offset in range(-4, 6)]
        if any(time not in matrix.index for time in times):
            continue
        local = matrix.loc[times, [BTC, *FROZEN_SYMBOLS]]
        if not bool(np.isfinite(local.to_numpy(dtype=float)).all()):
            continue
        direction = int(event.signal_direction)
        btc = local[BTC]
        gross_1h = direction * (
            float(btc.at[entry + pd.Timedelta(hours=1)]) / float(btc.at[entry]) - 1
        )
        gross_4h = direction * (
            float(btc.at[entry + pd.Timedelta(hours=4)]) / float(btc.at[entry]) - 1
        )
        delayed = direction * (
            float(btc.at[entry + pd.Timedelta(hours=5)])
            / float(btc.at[entry + pd.Timedelta(hours=1)])
            - 1
        )
        alt_gross = direction * float(
            (
                local.loc[entry + pd.Timedelta(hours=4), list(FROZEN_SYMBOLS)]
                / local.loc[entry, list(FROZEN_SYMBOLS)]
                - 1
            ).mean()
        )
        returns = btc.pct_change(fill_method=None)
        prior_variance = float(
            np.square(returns.loc[entry - pd.Timedelta(hours=3) : entry]).sum()
        )
        future_variance = float(
            np.square(
                returns.loc[
                    entry + pd.Timedelta(hours=1) : entry + pd.Timedelta(hours=4)
                ]
            ).sum()
        )
        rows.append(
            {
                "entry_time": entry,
                "btc_gross_return_1h": gross_1h,
                "btc_primary_net_return_1h": gross_1h - cfg.btc_primary_cost,
                "btc_gross_return_4h": gross_4h,
                "btc_primary_net_return_4h": gross_4h - cfg.btc_primary_cost,
                "btc_stress_net_return_4h": gross_4h - cfg.btc_stress_cost,
                "reversed_primary_net_return_4h": -gross_4h
                - cfg.btc_primary_cost,
                "delayed_gross_return_4h": delayed,
                "delayed_primary_net_return_4h": delayed - cfg.btc_primary_cost,
                "alt_bucket_gross_return_4h": alt_gross,
                "alt_bucket_primary_net_return_4h": alt_gross
                - cfg.alt_primary_cost,
                "alt_bucket_stress_net_return_4h": alt_gross - cfg.alt_stress_cost,
                "prior_btc_realized_variance_4h": prior_variance,
                "future_btc_realized_variance_4h": future_variance,
                "future_to_prior_btc_variance_ratio": future_variance / prior_variance,
            }
        )
    return pd.DataFrame(rows)


def _rebuild_no_vacuum(states: pd.DataFrame) -> pd.DataFrame:
    state = (
        states["bucket_pressure"].abs().ge(states["prior_abs_pressure_threshold"])
        & states["directional_symbol_count"].ge(11)
        & states["withdrawing_symbol_count"].lt(5)
    )
    starts = states[state & ~state.shift(1, fill_value=False)].copy()
    selected = []
    previous: pd.Timestamp | None = None
    for index, row in starts.iterrows():
        time = pd.Timestamp(row["decision_time"])
        if previous is None or time - previous >= pd.Timedelta(hours=4):
            selected.append(index)
            previous = time
    output = starts.loc[selected].copy()
    output["entry_time"] = output["decision_time"]
    output["signal_direction"] = output["direction"].astype(int)
    output["entry_month"] = output["entry_time"].dt.strftime("%Y-%m")
    output["period"] = np.select(
        [
            output["entry_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            output["entry_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    return output.reset_index(drop=True)


def _replay_random_controls(
    events: pd.DataFrame,
    states: pd.DataFrame,
    matrix: pd.DataFrame,
    cfg: V226Config,
) -> pd.DataFrame:
    candidate_times = set(events["entry_time"])
    pool = states[
        states["prior_abs_pressure_threshold"].notna()
        & ~states["decision_time"].isin(candidate_times)
    ].copy()
    pool["entry_time"] = pool["decision_time"]
    pool["entry_month"] = pool["entry_time"].dt.strftime("%Y-%m")
    pool["signal_direction"] = pool["direction"].astype(int)
    pool = pool[
        pool["entry_time"].map(
            lambda time: time in matrix.index
            and time + pd.Timedelta(hours=4) in matrix.index
        )
    ].copy()
    btc = matrix[BTC]
    pool["signed_gross"] = [
        int(row.signal_direction)
        * (float(btc.at[row.entry_time + pd.Timedelta(hours=4)]) / float(btc.at[row.entry_time]) - 1)
        for row in pool.itertuples(index=False)
    ]
    pools = {
        key: local["signed_gross"].to_numpy(dtype=float)
        for key, local in pool.groupby(
            ["entry_month", "signal_direction"], sort=False, observed=True
        )
    }
    requested = (
        events.groupby(["entry_month", "signal_direction"], observed=True)
        .size()
        .to_dict()
    )
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        values = []
        for key, count in requested.items():
            values.extend(rng.choice(pools[key], size=int(count), replace=True))
        mean_gross = float(np.mean(values))
        rows.append(
            {
                "iteration": iteration,
                "mean_gross_return_4h": mean_gross,
                "mean_primary_net_return_4h": mean_gross - cfg.btc_primary_cost,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(outcomes: pd.DataFrame, cfg: V226Config) -> tuple[float, float]:
    groups = [
        local["btc_primary_net_return_4h"].to_numpy(dtype=float)
        for _, local in outcomes.groupby("entry_day", sort=False, observed=True)
    ]
    rng = np.random.default_rng(cfg.seed + 2)
    draws = []
    for _ in range(cfg.bootstrap_iterations):
        indices = rng.integers(0, len(groups), size=len(groups))
        draws.append(float(np.concatenate([groups[index] for index in indices]).mean()))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _max_errors(left: pd.DataFrame, right: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    merged = left[["entry_time", *fields]].merge(
        right[["entry_time", *fields]],
        on="entry_time",
        suffixes=("_audit", "_saved"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        [
            {
                "field": field,
                "maximum_absolute_error": float(
                    (merged[f"{field}_audit"] - merged[f"{field}_saved"]).abs().max()
                ),
            }
            for field in fields
        ]
    )


def run_v226_audit(
    cfg: V226Config = V226Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_events = _utc(
        pd.read_parquet(cfg.v224_root / "candidate_feature_events.parquet")
    )
    states = _utc(pd.read_parquet(cfg.v224_root / "hourly_bucket_states.parquet"))
    saved = _utc(pd.read_parquet(cfg.v225_root / "candidate_events.parquet"))
    saved_no_vacuum = _utc(
        pd.read_parquet(cfg.v225_root / "no_vacuum_control_events.parquet")
    )
    saved_random = pd.read_csv(cfg.v225_root / "random_time_controls.csv")
    summary = pd.read_csv(cfg.v225_root / "candidate_outcome.csv").iloc[0]
    metadata = json.loads((cfg.v225_root / "metadata.json").read_text(encoding="utf-8"))
    matrix = _matrix()
    repriced = _reprice(feature_events, matrix, cfg)
    no_vacuum_features = _rebuild_no_vacuum(states)
    repriced_no_vacuum = _reprice(no_vacuum_features, matrix, cfg)
    random = _replay_random_controls(saved, states, matrix, cfg)
    fields = [
        "btc_gross_return_1h",
        "btc_primary_net_return_1h",
        "btc_gross_return_4h",
        "btc_primary_net_return_4h",
        "btc_stress_net_return_4h",
        "reversed_primary_net_return_4h",
        "delayed_gross_return_4h",
        "delayed_primary_net_return_4h",
        "alt_bucket_gross_return_4h",
        "alt_bucket_primary_net_return_4h",
        "alt_bucket_stress_net_return_4h",
        "prior_btc_realized_variance_4h",
        "future_btc_realized_variance_4h",
        "future_to_prior_btc_variance_ratio",
    ]
    candidate_errors = _max_errors(repriced, saved, fields)
    no_vacuum_errors = _max_errors(repriced_no_vacuum, saved_no_vacuum, fields)
    no_vacuum_errors["field"] = "no_vacuum_" + no_vacuum_errors["field"]
    random_merged = random.merge(
        saved_random[
            ["iteration", "mean_gross_return_4h", "mean_primary_net_return_4h"]
        ],
        on="iteration",
        suffixes=("_audit", "_saved"),
        validate="one_to_one",
    )
    random_errors = pd.DataFrame(
        [
            {
                "field": f"random_{field}",
                "maximum_absolute_error": float(
                    (random_merged[f"{field}_audit"] - random_merged[f"{field}_saved"])
                    .abs()
                    .max()
                ),
            }
            for field in ("mean_gross_return_4h", "mean_primary_net_return_4h")
        ]
    )
    low, high = _bootstrap(saved, cfg)
    periods = saved.groupby("period", observed=True)["btc_primary_net_return_4h"].mean()
    variances = saved.groupby("period", observed=True)[
        "future_to_prior_btc_variance_ratio"
    ].mean()
    directions = saved.groupby("signal_direction", observed=True)[
        "btc_primary_net_return_4h"
    ].mean()
    months = saved.groupby("entry_month", observed=True)[
        "btc_primary_net_return_4h"
    ].sum()
    days = saved.groupby("entry_day", observed=True)["btc_primary_net_return_4h"].sum()
    positive_months = months[months.gt(0)]
    positive_days = days[days.gt(0)]
    observed = float(saved["btc_primary_net_return_4h"].mean())
    expected = {
        "mean_btc_gross_1h_bp": saved["btc_gross_return_1h"].mean() * 10_000,
        "mean_btc_primary_net_1h_bp": saved["btc_primary_net_return_1h"].mean()
        * 10_000,
        "mean_btc_gross_4h_bp": saved["btc_gross_return_4h"].mean() * 10_000,
        "mean_btc_primary_net_4h_bp": observed * 10_000,
        "mean_btc_stress_net_4h_bp": saved["btc_stress_net_return_4h"].mean()
        * 10_000,
        "development_primary_net_4h_bp": periods.get("development", np.nan)
        * 10_000,
        "validation_primary_net_4h_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_4h_bp": periods.get("holdout", np.nan) * 10_000,
        "long_primary_net_4h_bp": directions.get(1, np.nan) * 10_000,
        "short_primary_net_4h_bp": directions.get(-1, np.nan) * 10_000,
        "mean_alt_bucket_gross_4h_bp": saved["alt_bucket_gross_return_4h"].mean()
        * 10_000,
        "mean_alt_bucket_primary_net_4h_bp": saved[
            "alt_bucket_primary_net_return_4h"
        ].mean()
        * 10_000,
        "mean_variance_ratio": saved["future_to_prior_btc_variance_ratio"].mean(),
        "validation_variance_ratio": variances.get("validation", np.nan),
        "holdout_variance_ratio": variances.get("holdout", np.nan),
        "bootstrap_95_low_primary_bp": low * 10_000,
        "bootstrap_95_high_primary_bp": high * 10_000,
        "random_time_percentile": 100
        * saved_random["mean_primary_net_return_4h"].le(observed).mean(),
        "reversed_primary_net_4h_bp": saved[
            "reversed_primary_net_return_4h"
        ].mean()
        * 10_000,
        "delayed_primary_net_4h_bp": saved["delayed_primary_net_return_4h"].mean()
        * 10_000,
        "no_vacuum_primary_net_4h_bp": saved_no_vacuum[
            "btc_primary_net_return_4h"
        ].mean()
        * 10_000,
        "positive_month_concentration": float(
            positive_months.max() / positive_months.sum()
        ),
        "positive_day_concentration": float(positive_days.max() / positive_days.sum()),
    }
    summary_errors = pd.DataFrame(
        [
            {
                "field": f"summary_{field}",
                "maximum_absolute_error": abs(float(summary[field]) - float(value)),
            }
            for field, value in expected.items()
        ]
    )
    errors = pd.concat(
        [candidate_errors, no_vacuum_errors, random_errors, summary_errors],
        ignore_index=True,
    )
    gates = {
        "coverage": summary["events"] >= 150
        and summary["active_months"] >= 11
        and summary["development_events"] >= 45
        and summary["validation_events"] >= 45
        and summary["holdout_events"] >= 45
        and summary["minimum_direction_period_events"] >= 15,
        "positive_1h_gross": summary["mean_btc_gross_1h_bp"] > 0,
        "positive_4h_gross": summary["mean_btc_gross_4h_bp"] > 0,
        "positive_primary": summary["mean_btc_primary_net_4h_bp"] > 0,
        "positive_stress": summary["mean_btc_stress_net_4h_bp"] > 0,
        "positive_development": summary["development_primary_net_4h_bp"] > 0,
        "positive_validation": summary["validation_primary_net_4h_bp"] > 0,
        "positive_holdout": summary["holdout_primary_net_4h_bp"] > 0,
        "positive_long": summary["long_primary_net_4h_bp"] > 0,
        "positive_short": summary["short_primary_net_4h_bp"] > 0,
        "positive_bootstrap_lower": summary["bootstrap_95_low_primary_bp"] > 0,
        "random_percentile_95": summary["random_time_percentile"] >= 95,
        "beats_reversed": summary["mean_btc_primary_net_4h_bp"]
        > summary["reversed_primary_net_4h_bp"],
        "beats_delayed": summary["mean_btc_primary_net_4h_bp"]
        > summary["delayed_primary_net_4h_bp"],
        "beats_no_vacuum": summary["mean_btc_primary_net_4h_bp"]
        > summary["no_vacuum_primary_net_4h_bp"],
        "variance_expands_all": summary["mean_variance_ratio"] > 1,
        "variance_expands_validation": summary["validation_variance_ratio"] > 1,
        "variance_expands_holdout": summary["holdout_variance_ratio"] > 1,
        "month_concentration": summary["positive_month_concentration"] <= 0.35,
        "day_concentration": summary["positive_day_concentration"] <= 0.20,
    }
    gate_frame = pd.DataFrame({"gate": list(gates), "passed": list(gates.values())})
    feature_hash = _sha256(cfg.v224_root / "candidate_feature_events.parquet")
    all_gates = bool(gate_frame["passed"].all())
    checks = {
        "v224_feature_checks_all_pass": bool(
            pd.read_csv(cfg.v224_root / "data_quality_checks.csv")["passed"].all()
        ),
        "feature_hash_matches_preregistration": feature_hash == FEATURE_SHA256,
        "feature_hash_matches_v225_metadata": feature_hash
        == metadata["feature_sha256"],
        "candidate_has_159_unique_events": len(saved) == 159
        and saved["entry_time"].is_unique,
        "holding_window_exactly_four_hours": bool(
            saved["exit_time"].sub(saved["entry_time"]).eq(pd.Timedelta(hours=4)).all()
        ),
        "four_hour_event_cooldown": bool(
            saved["entry_time"].sort_values().diff().dropna().ge(pd.Timedelta(hours=4)).all()
        ),
        "independent_candidate_repricing_exact": float(
            candidate_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "no_vacuum_feature_control_rebuilt_307": len(no_vacuum_features) == 307,
        "independent_no_vacuum_repricing_exact": float(
            no_vacuum_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "all_1000_random_paths_replayed_exactly": len(random) == 1000
        and float(random_errors["maximum_absolute_error"].max()) <= cfg.tolerance,
        "bootstrap_and_summary_exact": float(
            summary_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "promotion_flag_matches_gate_conjunction": bool(summary["promote"])
        == all_gates,
        "failed_gates_force_rejection": (not all_gates) and (not bool(summary["promote"])),
        "metadata_has_no_promotion": metadata["promoted"] == [],
        "metadata_records_no_permission_change": metadata["permissions_changed"] == [],
        "findings_records_rejection": "Verdict: `reject_alt_book_vacuum_pressure_to_btc`."
        in (
            cfg.v225_root.parent.parent
            / "docs/v225_alt_book_vacuum_pressure_to_btc_findings_2026_07_17.md"
        ).read_text(encoding="utf-8"),
    }
    audit_checks = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    hashes = pd.DataFrame(
        [
            {
                "artifact": str(cfg.v224_root / "candidate_feature_events.parquet"),
                "sha256": feature_hash,
            },
            {
                "artifact": str(cfg.v225_root / "candidate_events.parquet"),
                "sha256": _sha256(cfg.v225_root / "candidate_events.parquet"),
            },
            {
                "artifact": str(cfg.v225_root / "random_time_controls.csv"),
                "sha256": _sha256(cfg.v225_root / "random_time_controls.csv"),
            },
        ]
    )
    return audit_checks, errors, gate_frame, hashes


def write_v226_alt_book_vacuum_pressure_audit(
    cfg: V226Config = V226Config(),
) -> dict[str, Path]:
    checks, errors, gates, hashes = run_v226_audit(cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "checks": root / "independent_audit_checks.csv",
        "errors": root / "maximum_errors.csv",
        "gates": root / "promotion_gate_results.csv",
        "hashes": root / "artifact_hashes.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    checks.to_csv(paths["checks"], index=False)
    errors.to_csv(paths["errors"], index=False)
    gates.to_csv(paths["gates"], index=False)
    hashes.to_csv(paths["hashes"], index=False)
    passed = bool(checks["passed"].all())
    failed_gates = gates.loc[~gates["passed"], "gate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "audit_passed": passed,
                "checks_passed": int(checks["passed"].sum()),
                "checks_total": len(checks),
                "validated_verdict": (
                    "reject_alt_book_vacuum_pressure_to_btc"
                    if passed
                    else "audit_failed"
                ),
                "failed_promotion_gates": failed_gates,
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
                "# v22.6 Alt-Book Vacuum Pressure Independent Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "Failed promotion gates: "
                + ", ".join(f"`{gate}`" for gate in failed_gates),
                "",
                "Exact prices, all candidate/control returns, variance ratios, all",
                "1,000 random paths, bootstrap summaries and governance metadata were",
                "independently reconstructed. The rejection is valid.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
