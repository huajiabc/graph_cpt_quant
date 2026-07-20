"""Independent audit of the v23.1 synthetic straddle diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.deribit_option_trade_history import inverse_option_price
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    load_v155_hourly_prices,
)
from pressure_graph.reports.v231_book_vacuum_synthetic_straddle import (
    FEATURE_SHA256,
    YEAR_HOURS,
)


V230_ROOT = Path("reports/v23_0_book_vacuum_implied_variance_feature_audit")
V231_ROOT = Path("reports/v23_1_book_vacuum_synthetic_straddle")
RAW_EVENT_PATH = Path(
    "reports/v22_4_alt_book_vacuum_pressure_feature_audit/"
    "candidate_feature_events.parquet"
)
REPORT_ROOT = Path("reports/v23_2_book_vacuum_synthetic_straddle_audit")
FINDINGS_PATH = Path(
    "docs/v232_book_vacuum_synthetic_straddle_audit_2026_07_17.md"
)


@dataclass(frozen=True)
class V232Config:
    v230_root: Path = V230_ROOT
    v231_root: Path = V231_ROOT
    raw_event_path: Path = RAW_EVENT_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    primary_hurdle: float = 0.01
    stress_hurdle: float = 0.02
    horizons: tuple[int, ...] = (1, 4, 8)
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
        if column.endswith("_time") or column in {
            "surface_expiration_time",
            "surface_feature_time",
        }:
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def _close() -> pd.Series:
    prices = load_v155_hourly_prices()
    prices["feature_time"] = pd.to_datetime(
        prices["feature_time"], utc=True, errors="coerce"
    )
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return (
        prices[prices["symbol"].eq("BTCUSDT")]
        .dropna(subset=["feature_time", "close"])
        .drop_duplicates("feature_time", keep="last")
        .set_index("feature_time")["close"]
        .sort_index()
    )


def _formula_errors(saved: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    hourly_log_move = np.log(close / close.shift(1))
    errors: dict[str, float] = {}
    recomputed: dict[str, list[float]] = {}
    for horizon in (1, 4, 8):
        for field in (
            "exit_spot",
            "absolute_log_move",
            "realized_variance",
            "implied_variance_budget",
            "realized_to_implied_variance",
            "synthetic_exit_premium_btc",
            "synthetic_exit_premium_usd",
            "gross_premium_return",
        ):
            recomputed[f"{field}_{horizon}h"] = []
    recomputed["synthetic_entry_premium_btc"] = []
    recomputed["synthetic_entry_premium_usd"] = []
    recomputed["primary_net_premium_return_4h"] = []
    recomputed["stress_net_premium_return_4h"] = []
    for row in saved.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        spot = float(row.entry_spot)
        strike = float(row.synthetic_strike)
        expiry = pd.Timestamp(row.surface_expiration_time)
        volatility = float(row.causal_atm_iv)
        years = (expiry - entry).total_seconds() / (YEAR_HOURS * 3600.0)
        entry_btc = inverse_option_price(
            spot, strike, years, volatility, "call"
        ) + inverse_option_price(spot, strike, years, volatility, "put")
        entry_usd = entry_btc * spot
        recomputed["synthetic_entry_premium_btc"].append(entry_btc)
        recomputed["synthetic_entry_premium_usd"].append(entry_usd)
        gross4 = np.nan
        for horizon in (1, 4, 8):
            exit_time = entry + pd.Timedelta(hours=horizon)
            exit_spot = float(close.loc[exit_time])
            exit_years = years - horizon / YEAR_HOURS
            exit_btc = inverse_option_price(
                exit_spot, strike, exit_years, volatility, "call"
            ) + inverse_option_price(
                exit_spot, strike, exit_years, volatility, "put"
            )
            exit_usd = exit_btc * exit_spot
            gross = exit_usd / entry_usd - 1.0
            path = [
                entry + pd.Timedelta(hours=offset)
                for offset in range(1, horizon + 1)
            ]
            realized = float(np.square(hourly_log_move.loc[path]).sum())
            implied = volatility * volatility * horizon / YEAR_HOURS
            values = {
                f"exit_spot_{horizon}h": exit_spot,
                f"absolute_log_move_{horizon}h": abs(np.log(exit_spot / spot)),
                f"realized_variance_{horizon}h": realized,
                f"implied_variance_budget_{horizon}h": implied,
                f"realized_to_implied_variance_{horizon}h": realized / implied,
                f"synthetic_exit_premium_btc_{horizon}h": exit_btc,
                f"synthetic_exit_premium_usd_{horizon}h": exit_usd,
                f"gross_premium_return_{horizon}h": gross,
            }
            for field, value in values.items():
                recomputed[field].append(value)
            if horizon == 4:
                gross4 = gross
        recomputed["primary_net_premium_return_4h"].append(
            gross4 - 0.01
        )
        recomputed["stress_net_premium_return_4h"].append(
            gross4 - 0.02
        )
    for field, values in recomputed.items():
        errors[field] = float(
            np.max(np.abs(saved[field].to_numpy(dtype=float) - np.asarray(values)))
        )
    return pd.DataFrame(
        {
            "field": list(errors),
            "maximum_absolute_error": list(errors.values()),
        }
    )


def _expected_pool_pairs(
    features: pd.DataFrame,
    universe: pd.DataFrame,
    nearest: int,
) -> set[tuple[pd.Timestamp, pd.Timestamp]]:
    pairs: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    epsilon = 1e-12
    for event in features.itertuples(index=False):
        local = universe[
            universe["entry_month"].eq(event.entry_month)
            & universe["utc_hour"].eq(event.entry_time.hour)
        ].copy()
        local["distance"] = (
            np.log(local["causal_atm_iv"] / event.causal_atm_iv).abs()
            + np.log(
                (local["prior_24h_sum_squared_log_move"] + epsilon)
                / (event.prior_24h_sum_squared_log_move + epsilon)
            ).abs()
        )
        local = local.sort_values(["distance", "entry_time"]).head(nearest)
        pairs.update((event.entry_time, time) for time in local["entry_time"])
    return pairs


def _replay_random(
    outcomes: pd.DataFrame,
    controls: pd.DataFrame,
    pools: pd.DataFrame,
    cfg: V232Config,
) -> pd.DataFrame:
    field = "primary_net_premium_return_4h"
    event_lookup = outcomes.set_index("entry_time")[field]
    control_lookup = controls.set_index("entry_time")[field]
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
        mean = float(control_lookup.loc[sampled].mean())
        rows.append((iteration, event_mean, mean, event_mean - mean))
    return pd.DataFrame(
        rows,
        columns=[
            "iteration",
            "event_mean_primary_net",
            "control_mean_primary_net",
            "event_minus_control",
        ],
    )


def _replay_bootstrap(outcomes: pd.DataFrame, cfg: V232Config) -> pd.DataFrame:
    groups = {
        month: local["primary_net_premium_return_4h"].to_numpy(dtype=float)
        for month, local in outcomes.groupby("entry_month", sort=True)
    }
    months = sorted(groups)
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
        sampled = rng.choice(months, size=len(months), replace=True)
        mean = float(np.concatenate([groups[month] for month in sampled]).mean())
        rows.append((iteration, mean))
    return pd.DataFrame(
        rows,
        columns=["iteration", "mean_primary_net_premium_return"],
    )


def run_v232_audit(
    cfg: V232Config = V232Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _utc(
        pd.read_parquet(cfg.v230_root / "causal_implied_variance_features.parquet")
    )
    outcomes = _utc(
        pd.read_parquet(cfg.v231_root / "synthetic_straddle_event_outcomes.parquet")
    )
    universe = _utc(pd.read_parquet(cfg.v231_root / "causal_control_universe.parquet"))
    pools = _utc(pd.read_parquet(cfg.v231_root / "matched_control_pools.parquet"))
    controls = _utc(
        pd.read_parquet(cfg.v231_root / "synthetic_straddle_control_outcomes.parquet")
    )
    saved_random = pd.read_parquet(cfg.v231_root / "matched_random_paths.parquet")
    saved_bootstrap = pd.read_parquet(cfg.v231_root / "month_block_bootstrap.parquet")
    saved_summary = pd.read_csv(cfg.v231_root / "result_summary.csv")
    saved_decision = pd.read_csv(cfg.v231_root / "decision_gates.csv")
    close = _close()
    errors = _formula_errors(outcomes, close)

    expected_pairs = _expected_pool_pairs(features, universe, cfg.nearest_controls)
    saved_pairs = set(zip(pools["event_time"], pools["control_time"], strict=True))
    raw_events = _utc(pd.read_parquet(cfg.raw_event_path))
    minimum_distance = min(
        abs((control - event).total_seconds()) / 3600.0
        for control in universe["entry_time"]
        for event in raw_events["entry_time"]
        if abs((control - event).total_seconds()) / 3600.0 <= 24.0
    )
    replay_random = _replay_random(outcomes, controls, pools, cfg)
    replay_bootstrap = _replay_bootstrap(outcomes, cfg)
    random_error = float(
        np.max(
            np.abs(
                replay_random[
                    [
                        "event_mean_primary_net",
                        "control_mean_primary_net",
                        "event_minus_control",
                    ]
                ].to_numpy()
                - saved_random[
                    [
                        "event_mean_primary_net",
                        "control_mean_primary_net",
                        "event_minus_control",
                    ]
                ].to_numpy()
            )
        )
    )
    bootstrap_error = float(
        np.max(
            np.abs(
                replay_bootstrap["mean_primary_net_premium_return"].to_numpy()
                - saved_bootstrap["mean_primary_net_premium_return"].to_numpy()
            )
        )
    )

    all_row = saved_summary[saved_summary["scope"].eq("all")].iloc[0]
    period_min = float(
        saved_summary[
            saved_summary["scope"].isin(["development", "validation", "holdout"])
        ]["mean_primary_net_premium_return_4h_bp"].min()
    )
    lower = float(
        saved_bootstrap["mean_primary_net_premium_return"].quantile(0.025)
        * 10_000
    )
    event_mean = float(saved_random["event_mean_primary_net"].iloc[0])
    percentile = float(
        saved_random["control_mean_primary_net"].le(event_mean).mean() * 100
    )
    observed = np.asarray(
        [
            all_row["mean_primary_net_premium_return_4h_bp"],
            period_min,
            lower,
            percentile,
            all_row["mean_realized_to_implied_variance_4h"],
        ],
        dtype=float,
    )
    decision_error = float(
        np.max(np.abs(observed - saved_decision["observed"].to_numpy(dtype=float)))
    )

    checks = {
        "v230_feature_audit_passed": bool(
            pd.read_csv(cfg.v230_root / "data_quality_checks.csv")["passed"].all()
        ),
        "feature_hash_matches_preregistration": _sha256(
            cfg.v230_root / "causal_implied_variance_features.parquet"
        )
        == FEATURE_SHA256,
        "event_keys_match_feature_freeze": set(outcomes["entry_time"])
        == set(features["entry_time"]),
        "all_inverse_option_and_path_formulas_exact": float(
            errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "primary_and_stress_hurdles_exact": bool(
            np.allclose(
                outcomes["primary_net_premium_return_4h"],
                outcomes["gross_premium_return_4h"] - cfg.primary_hurdle,
                atol=cfg.tolerance,
            )
            and np.allclose(
                outcomes["stress_net_premium_return_4h"],
                outcomes["gross_premium_return_4h"] - cfg.stress_hurdle,
                atol=cfg.tolerance,
            )
        ),
        "matched_controls_share_month_and_utc_hour": bool(
            pools["event_month"].eq(pools["control_time"].dt.strftime("%Y-%m")).all()
            and pools["event_time"].dt.hour.eq(pools["control_time"].dt.hour).all()
        ),
        "control_event_exclusion_exceeds_8h": minimum_distance
        > cfg.event_exclusion_hours,
        "nearest_10_control_pairs_exact": saved_pairs == expected_pairs,
        "all_123_events_have_frozen_5_to_10_controls": pools[
            "event_time"
        ].nunique()
        == 123
        and pools.groupby("event_time").size().between(5, cfg.nearest_controls).all(),
        "all_1000_random_paths_replayed_exactly": len(saved_random)
        == cfg.random_iterations
        and random_error <= cfg.tolerance,
        "all_5000_month_bootstraps_replayed_exactly": len(saved_bootstrap)
        == cfg.bootstrap_iterations
        and bootstrap_error <= cfg.tolerance,
        "decision_observations_recomputed_exactly": decision_error <= cfg.tolerance,
        "failed_absolute_gates_force_rejection": bool(
            (~saved_decision["passed"]).any()
            and not bool(
                saved_decision.loc[
                    saved_decision["gate"].eq("overall_primary_net_positive"),
                    "passed",
                ].iloc[0]
            )
        ),
        "matched_relative_edge_and_variance_edge_preserved": bool(
            saved_decision.loc[
                saved_decision["gate"].isin(
                    [
                        "matched_random_percentile_at_least_90",
                        "mean_realized_to_implied_variance_above_one",
                    ]
                ),
                "passed",
            ].all()
        ),
        "findings_record_research_only_rejection": (
            "Verdict: `movement_sufficiency_rejected`."
            in (
                cfg.v231_root.parent.parent
                / "docs/v231_book_vacuum_synthetic_straddle_findings_2026_07_17.md"
            ).read_text(encoding="utf-8")
        ),
    }
    audit_checks = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "minimum_control_event_distance_hours", "value": minimum_distance},
            {"diagnostic": "random_maximum_error", "value": random_error},
            {"diagnostic": "bootstrap_maximum_error", "value": bootstrap_error},
            {"diagnostic": "decision_maximum_error", "value": decision_error},
            {"diagnostic": "matched_random_percentile", "value": percentile},
            {"diagnostic": "bootstrap_lower_bp", "value": lower},
        ]
    )
    return audit_checks, errors, diagnostics


def write_v232_book_vacuum_synthetic_straddle_audit(
    cfg: V232Config = V232Config(),
) -> dict[str, Path]:
    checks, errors, diagnostics = run_v232_audit(cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "checks": root / "independent_audit_checks.csv",
        "errors": root / "maximum_formula_errors.csv",
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
                "validated_verdict": (
                    "movement_sufficiency_rejected" if passed else "audit_failed"
                ),
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
                "# v23.2 Synthetic Straddle Independent Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "Inverse-option repricing, BTC paths, implied-variance budgets,",
                "control matching, 1,000 random paths, 5,000 month bootstraps,",
                "decision gates, and the research-only rejection were replayed.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V232Config",
    "run_v232_audit",
    "write_v232_book_vacuum_synthetic_straddle_audit",
]
