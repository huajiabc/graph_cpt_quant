"""Independent audit of the v22.8 vacuum-pressure rank spread."""

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


V227_ROOT = Path("reports/v22_7_vacuum_pressure_cross_section_feature_audit")
V228_ROOT = Path("reports/v22_8_vacuum_pressure_cross_section_spread")
REPORT_ROOT = Path("reports/v22_9_vacuum_pressure_cross_section_audit")
FINDINGS_PATH = Path(
    "docs/v229_vacuum_pressure_cross_section_audit_2026_07_17.md"
)
FEATURE_SHA256 = "5B7D351886C63DA3178B81C101BF24A47CA90085651A6654B414226674DD546E"


@dataclass(frozen=True)
class V229Config:
    v227_root: Path = V227_ROOT
    v228_root: Path = V228_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    primary_cost: float = 0.0030
    stress_cost: float = 0.0040
    raw_cost: float = 0.0020
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
        "month_start",
        "history_start",
        "history_end_exclusive",
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


def _recompute_betas(features: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    returns = matrix.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    rows = []
    for month in sorted(features["entry_month"].unique()):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        history = returns[
            (returns.index >= start - pd.Timedelta(days=30))
            & (returns.index < start)
        ]
        for symbol in FROZEN_SYMBOLS:
            paired = history[[symbol, BTC]].dropna()
            btc = paired[BTC].to_numpy(dtype=float)
            alt = paired[symbol].to_numpy(dtype=float)
            variance = float(np.var(btc, ddof=0))
            covariance = float(
                np.mean((alt - alt.mean()) * (btc - btc.mean()))
            )
            rows.append(
                {
                    "entry_month": month,
                    "symbol": symbol,
                    "beta_samples": len(paired),
                    "btc_beta": covariance / variance,
                }
            )
    return pd.DataFrame(rows)


def _weights(
    raw: dict[str, float], beta_map: dict[str, float]
) -> dict[str, float]:
    hedge = -sum(raw[symbol] * beta_map[symbol] for symbol in raw)
    unscaled = {**raw, BTC: hedge}
    gross = sum(abs(value) for value in unscaled.values())
    return {symbol: value / gross for symbol, value in unscaled.items()}


def _recompute_paths(
    features: pd.DataFrame,
    betas: pd.DataFrame,
    matrix: pd.DataFrame,
    cfg: V229Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_maps = {
        month: dict(zip(local["symbol"], local["btc_beta"], strict=True))
        for month, local in betas.groupby("entry_month", observed=True)
    }
    rows = []
    weight_rows = []
    for entry, local in features.groupby("entry_time", sort=True, observed=True):
        entry = pd.Timestamp(entry)
        month = str(local["entry_month"].iloc[0])
        beta_map = beta_maps[month]
        raw = dict(zip(local["symbol"], local["raw_weight"], strict=True))
        weights = _weights(raw, beta_map)
        returns = {
            horizon: {
                symbol: float(
                    matrix.at[entry + pd.Timedelta(hours=horizon), symbol]
                    / matrix.at[entry, symbol]
                    - 1
                )
                for symbol in weights
            }
            for horizon in (1, 4, 8)
        }
        gross = {
            horizon: float(
                sum(weights[symbol] * values[symbol] for symbol in weights)
            )
            for horizon, values in returns.items()
        }
        delayed = float(
            sum(
                weights[symbol]
                * (
                    matrix.at[entry + pd.Timedelta(hours=5), symbol]
                    / matrix.at[entry + pd.Timedelta(hours=1), symbol]
                    - 1
                )
                for symbol in weights
            )
        )
        raw_gross = float(sum(raw[symbol] * returns[4][symbol] for symbol in raw))
        residual = float(
            sum(weights[symbol] * beta_map[symbol] for symbol in raw) + weights[BTC]
        )
        rows.append(
            {
                "entry_time": entry,
                "gross_return_1h": gross[1],
                "gross_return_4h": gross[4],
                "gross_return_8h": gross[8],
                "primary_net_return_4h": gross[4] - cfg.primary_cost,
                "stress_net_return_4h": gross[4] - cfg.stress_cost,
                "reversed_primary_net_return_4h": -gross[4] - cfg.primary_cost,
                "delayed_gross_return_4h": delayed,
                "delayed_primary_net_return_4h": delayed - cfg.primary_cost,
                "raw_dollar_neutral_gross_return_4h": raw_gross,
                "raw_dollar_neutral_net_return_4h": raw_gross - cfg.raw_cost,
                "long_alt_return_4h": float(
                    sum(
                        weights[symbol] * returns[4][symbol]
                        for symbol in raw
                        if raw[symbol] > 0
                    )
                ),
                "short_alt_return_4h": float(
                    sum(
                        weights[symbol] * returns[4][symbol]
                        for symbol in raw
                        if raw[symbol] < 0
                    )
                ),
                "btc_hedge_return_4h": weights[BTC] * returns[4][BTC],
                "btc_hedge_weight": weights[BTC],
                "residual_btc_beta": residual,
                "gross_notional": float(sum(abs(value) for value in weights.values())),
            }
        )
        weight_rows.extend(
            {
                "entry_time": entry,
                "symbol": symbol,
                "weight": weight,
                "btc_beta": 1.0 if symbol == BTC else beta_map[symbol],
            }
            for symbol, weight in weights.items()
        )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def _random_paths(
    saved: pd.DataFrame,
    betas: pd.DataFrame,
    matrix: pd.DataFrame,
    cfg: V229Config,
) -> pd.DataFrame:
    beta_maps = {
        month: dict(zip(local["symbol"], local["btc_beta"], strict=True))
        for month, local in betas.groupby("entry_month", observed=True)
    }
    symbols = np.asarray(sorted(FROZEN_SYMBOLS))
    contexts = []
    for event in saved.itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        future = entry + pd.Timedelta(hours=4)
        returns = {
            symbol: float(matrix.at[future, symbol] / matrix.at[entry, symbol] - 1)
            for symbol in [BTC, *symbols]
        }
        contexts.append((beta_maps[str(event.entry_month)], returns))
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        returns = []
        for beta_map, future_returns in contexts:
            order = rng.permutation(symbols)
            raw = {symbol: 0.125 for symbol in order[:4]}
            raw.update({symbol: -0.125 for symbol in order[-4:]})
            weights = _weights(raw, beta_map)
            gross = sum(
                weights[symbol] * future_returns[symbol] for symbol in weights
            )
            returns.append(gross - cfg.primary_cost)
        rows.append(
            {
                "iteration": iteration,
                "mean_primary_net_return_4h": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(saved: pd.DataFrame, cfg: V229Config) -> tuple[float, float]:
    groups = [
        local["primary_net_return_4h"].to_numpy(dtype=float)
        for _, local in saved.groupby("entry_day", sort=False, observed=True)
    ]
    rng = np.random.default_rng(cfg.seed + 2)
    draws = []
    for _ in range(cfg.bootstrap_iterations):
        indices = rng.integers(0, len(groups), size=len(groups))
        draws.append(float(np.concatenate([groups[index] for index in indices]).mean()))
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def _errors(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    fields: list[str],
    prefix: str,
) -> pd.DataFrame:
    merged = left[[*keys, *fields]].merge(
        right[[*keys, *fields]],
        on=keys,
        suffixes=("_audit", "_saved"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        [
            {
                "field": f"{prefix}{field}",
                "maximum_absolute_error": float(
                    (merged[f"{field}_audit"] - merged[f"{field}_saved"]).abs().max()
                ),
            }
            for field in fields
        ]
    )


def run_v229_audit(
    cfg: V229Config = V229Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _utc(pd.read_parquet(cfg.v227_root / "ranked_symbol_features.parquet"))
    saved = _utc(pd.read_parquet(cfg.v228_root / "candidate_events.parquet"))
    saved_weights = _utc(pd.read_parquet(cfg.v228_root / "event_weights.parquet"))
    saved_betas = _utc(pd.read_parquet(cfg.v228_root / "monthly_betas.parquet"))
    saved_random = pd.read_csv(cfg.v228_root / "random_rank_controls.csv")
    summary = pd.read_csv(cfg.v228_root / "candidate_outcome.csv").iloc[0]
    metadata = json.loads((cfg.v228_root / "metadata.json").read_text(encoding="utf-8"))
    matrix = _matrix()
    betas = _recompute_betas(features, matrix)
    paths, weights = _recompute_paths(features, betas, matrix, cfg)
    random = _random_paths(saved, betas, matrix, cfg)
    beta_errors = _errors(
        betas,
        saved_betas,
        ["entry_month", "symbol"],
        ["beta_samples", "btc_beta"],
        "beta_",
    )
    weight_errors = _errors(
        weights,
        saved_weights,
        ["entry_time", "symbol"],
        ["weight", "btc_beta"],
        "weight_",
    )
    path_fields = [
        "gross_return_1h",
        "gross_return_4h",
        "gross_return_8h",
        "primary_net_return_4h",
        "stress_net_return_4h",
        "reversed_primary_net_return_4h",
        "delayed_gross_return_4h",
        "delayed_primary_net_return_4h",
        "raw_dollar_neutral_gross_return_4h",
        "raw_dollar_neutral_net_return_4h",
        "long_alt_return_4h",
        "short_alt_return_4h",
        "btc_hedge_return_4h",
        "btc_hedge_weight",
        "residual_btc_beta",
        "gross_notional",
    ]
    path_errors = _errors(paths, saved, ["entry_time"], path_fields, "path_")
    random_errors = _errors(
        random,
        saved_random,
        ["iteration"],
        ["mean_primary_net_return_4h"],
        "random_",
    )
    low, high = _bootstrap(saved, cfg)
    periods = saved.groupby("period", observed=True)["primary_net_return_4h"].mean()
    months = saved.groupby("entry_month", observed=True)["primary_net_return_4h"].sum()
    days = saved.groupby("entry_day", observed=True)["primary_net_return_4h"].sum()
    positive_months = months[months.gt(0)]
    positive_days = days[days.gt(0)]
    observed = float(saved["primary_net_return_4h"].mean())
    expected = {
        "mean_gross_1h_bp": saved["gross_return_1h"].mean() * 10_000,
        "mean_gross_4h_bp": saved["gross_return_4h"].mean() * 10_000,
        "mean_gross_8h_bp": saved["gross_return_8h"].mean() * 10_000,
        "mean_primary_net_4h_bp": observed * 10_000,
        "mean_stress_net_4h_bp": saved["stress_net_return_4h"].mean() * 10_000,
        "development_primary_net_4h_bp": periods.get("development", np.nan)
        * 10_000,
        "validation_primary_net_4h_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_4h_bp": periods.get("holdout", np.nan) * 10_000,
        "raw_dollar_neutral_gross_4h_bp": saved[
            "raw_dollar_neutral_gross_return_4h"
        ].mean()
        * 10_000,
        "raw_dollar_neutral_net_4h_bp": saved[
            "raw_dollar_neutral_net_return_4h"
        ].mean()
        * 10_000,
        "long_alt_component_4h_bp": saved["long_alt_return_4h"].mean() * 10_000,
        "short_alt_component_4h_bp": saved["short_alt_return_4h"].mean() * 10_000,
        "btc_hedge_component_4h_bp": saved["btc_hedge_return_4h"].mean() * 10_000,
        "bootstrap_95_low_primary_bp": low * 10_000,
        "bootstrap_95_high_primary_bp": high * 10_000,
        "random_rank_percentile": 100
        * saved_random["mean_primary_net_return_4h"].le(observed).mean(),
        "reversed_primary_net_4h_bp": saved[
            "reversed_primary_net_return_4h"
        ].mean()
        * 10_000,
        "delayed_primary_net_4h_bp": saved["delayed_primary_net_return_4h"].mean()
        * 10_000,
        "positive_month_concentration": (
            float(positive_months.max() / positive_months.sum())
            if positive_months.sum() > 0
            else np.inf
        ),
        "positive_day_concentration": float(positive_days.max() / positive_days.sum()),
        "max_abs_residual_btc_beta": saved["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (saved["gross_notional"] - 1).abs().max(),
    }
    summary_errors = pd.DataFrame(
        [
            {
                "field": f"summary_{field}",
                "maximum_absolute_error": (
                    0.0
                    if np.isinf(float(summary[field])) and np.isinf(float(value))
                    else abs(float(summary[field]) - float(value))
                ),
            }
            for field, value in expected.items()
        ]
    )
    errors = pd.concat(
        [beta_errors, weight_errors, path_errors, random_errors, summary_errors],
        ignore_index=True,
    )
    gates = {
        "coverage": summary["events"] >= 150
        and summary["active_months"] >= 11
        and summary["development_events"] >= 45
        and summary["validation_events"] >= 45
        and summary["holdout_events"] >= 45,
        "positive_gross": summary["mean_gross_4h_bp"] > 0,
        "positive_primary": summary["mean_primary_net_4h_bp"] > 0,
        "positive_stress": summary["mean_stress_net_4h_bp"] > 0,
        "positive_development": summary["development_primary_net_4h_bp"] > 0,
        "positive_validation": summary["validation_primary_net_4h_bp"] > 0,
        "positive_holdout": summary["holdout_primary_net_4h_bp"] > 0,
        "positive_raw_net": summary["raw_dollar_neutral_net_4h_bp"] > 0,
        "positive_bootstrap_lower": summary["bootstrap_95_low_primary_bp"] > 0,
        "random_percentile_95": summary["random_rank_percentile"] >= 95,
        "beats_reversed": summary["mean_primary_net_4h_bp"]
        > summary["reversed_primary_net_4h_bp"],
        "beats_delayed": summary["mean_primary_net_4h_bp"]
        > summary["delayed_primary_net_4h_bp"],
        "month_concentration": summary["positive_month_concentration"] <= 0.35,
        "day_concentration": summary["positive_day_concentration"] <= 0.20,
        "beta_exact": summary["max_abs_residual_btc_beta"] <= cfg.tolerance,
        "gross_exact": summary["max_gross_notional_drift"] <= cfg.tolerance,
    }
    gate_frame = pd.DataFrame({"gate": list(gates), "passed": list(gates.values())})
    feature_hash = _sha256(cfg.v227_root / "ranked_symbol_features.parquet")
    all_gates = bool(gate_frame["passed"].all())
    checks = {
        "v227_feature_checks_all_pass": bool(
            pd.read_csv(cfg.v227_root / "data_quality_checks.csv")["passed"].all()
        ),
        "feature_hash_matches_preregistration": feature_hash == FEATURE_SHA256,
        "feature_hash_matches_metadata": feature_hash == metadata["feature_sha256"],
        "monthly_betas_use_at_least_500_prior_samples": bool(
            saved_betas["beta_samples"].ge(500).all()
            and saved_betas["history_end_exclusive"].eq(saved_betas["month_start"]).all()
        ),
        "monthly_betas_recomputed_exactly": float(
            beta_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "event_weights_recomputed_exactly": float(
            weight_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "event_returns_and_costs_recomputed_exactly": float(
            path_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "gross_and_beta_exact": bool(
            saved["gross_notional"].sub(1).abs().le(cfg.tolerance).all()
            and saved["residual_btc_beta"].abs().le(cfg.tolerance).all()
        ),
        "all_1000_random_rank_paths_replayed_exactly": len(random) == 1000
        and float(random_errors["maximum_absolute_error"].max()) <= cfg.tolerance,
        "bootstrap_and_summary_recomputed_exactly": float(
            summary_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "promotion_flag_matches_gate_conjunction": bool(summary["promote"])
        == all_gates,
        "failed_gates_force_rejection": (not all_gates) and (not bool(summary["promote"])),
        "metadata_has_no_promotion": metadata["promoted"] == [],
        "metadata_records_no_permission_change": metadata["permissions_changed"] == [],
        "findings_records_rejection": "Verdict: `reject_vacuum_pressure_cross_section_spread`."
        in (
            cfg.v228_root.parent.parent
            / "docs/v228_vacuum_pressure_cross_section_spread_findings_2026_07_17.md"
        ).read_text(encoding="utf-8"),
    }
    audit_checks = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    hashes = pd.DataFrame(
        [
            {
                "artifact": str(cfg.v227_root / "ranked_symbol_features.parquet"),
                "sha256": feature_hash,
            },
            {
                "artifact": str(cfg.v228_root / "candidate_events.parquet"),
                "sha256": _sha256(cfg.v228_root / "candidate_events.parquet"),
            },
            {
                "artifact": str(cfg.v228_root / "random_rank_controls.csv"),
                "sha256": _sha256(cfg.v228_root / "random_rank_controls.csv"),
            },
        ]
    )
    return audit_checks, errors, gate_frame, hashes


def write_v229_vacuum_pressure_cross_section_audit(
    cfg: V229Config = V229Config(),
) -> dict[str, Path]:
    checks, errors, gates, hashes = run_v229_audit(cfg)
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
                    "reject_vacuum_pressure_cross_section_spread"
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
                "# v22.9 Vacuum-Pressure Cross-Section Independent Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "Failed gates: " + ", ".join(f"`{gate}`" for gate in failed_gates),
                "",
                "Betas, weights, all horizons, controls, 1,000 random paths,",
                "bootstrap summaries and governance metadata were independently rebuilt.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
