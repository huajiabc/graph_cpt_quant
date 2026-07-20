"""Independent arithmetic and governance audit of the v22.2 SFI overlay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    _moving_block_means,
)
from pressure_graph.reports.v147_funding_sign_spread import beta_neutral_components
from pressure_graph.reports.v151_causal_risk_parity_fss3_tg1 import (
    _additive_max_drawdown,
)


PANEL_PATH = Path(
    "reports/v13_4_negative_funding_beta_neutral_rebound/weekly_symbol_panel.parquet"
)
FEATURE_PATH = Path(
    "reports/v22_1_sfi_fss3_overlay_feature_audit/"
    "weekly_symbol_overlay_features.parquet"
)
FEATURE_CHECKS_PATH = Path(
    "reports/v22_1_sfi_fss3_overlay_feature_audit/data_quality_checks.csv"
)
V222_ROOT = Path("reports/v22_2_sfi_fss3_overlay")
FSS3_PATH = Path("reports/v14_9_funding_sign_turnover_cap/weekly_portfolio.parquet")
TG1_PATH = Path("reports/v13_2_tg1_forward_temporal_extension/weekly_portfolio.parquet")
CM2_PATH = Path(
    "reports/v16_5_fixed_core_satellite_fss3_tg1/weekly_portfolio.parquet"
)
REPORT_ROOT = Path("reports/v22_3_sfi_fss3_overlay_audit")
FINDINGS_PATH = Path("docs/v223_sfi_fss3_overlay_audit_2026_07_17.md")
FEATURE_SHA256 = "D1C10E394B0DF5D3202FB4342C38933ABB2D2137D8EB706FCC9570AE0F027D04"


@dataclass(frozen=True)
class V223Config:
    panel_path: Path = PANEL_PATH
    feature_path: Path = FEATURE_PATH
    feature_checks_path: Path = FEATURE_CHECKS_PATH
    v222_root: Path = V222_ROOT
    fss3_path: Path = FSS3_PATH
    tg1_path: Path = TG1_PATH
    cm2_path: Path = CM2_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    turnover_cap: float = 0.70
    fss3_weight: float = 0.80
    tg1_weight: float = 0.20
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260717
    tolerance: float = 1e-12
    cap_tolerance: float = 1e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("entry_time", "exit_time", "month_start", "sfi_feature_time"):
        if column in output:
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def _turnover(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        sum(
            abs(left.get(symbol, 0.0) - right.get(symbol, 0.0))
            for symbol in set(left) | set(right)
        )
    )


def _target(
    local: pd.DataFrame,
    week_features: pd.DataFrame | None,
) -> dict[str, float]:
    eligible = local.dropna(
        subset=["score_7d", "price_return", "future_funding", "btc_beta"]
    )
    longs = sorted(
        eligible.loc[eligible["score_7d"].lt(0), "symbol"].astype(str).unique()
    )
    shorts = sorted(
        eligible.loc[eligible["score_7d"].gt(0), "symbol"].astype(str).unique()
    )
    if week_features is None:
        raw = {symbol: 0.5 / len(longs) for symbol in longs}
        raw.update({symbol: -0.5 / len(shorts) for symbol in shorts})
    else:
        indexed = week_features.set_index("symbol", verify_integrity=True)
        raw = {
            symbol: float(indexed.at[symbol, "overlay_raw_weight"])
            for symbol in longs + shorts
        }
    weights, _ = beta_neutral_components(local, raw)
    return weights


def independently_recompute_overlay(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    portfolio: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: V223Config = V223Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_groups = {
        pd.Timestamp(entry): local.set_index("symbol", drop=False)
        for entry, local in panel.groupby("entry_time", sort=False, observed=True)
    }
    feature_groups = {
        pd.Timestamp(entry): local
        for entry, local in features.groupby("entry_time", sort=False, observed=True)
    }
    weight_groups = {
        pd.Timestamp(entry): dict(zip(local["symbol"], local["weight"], strict=True))
        for entry, local in weights.groupby("entry_time", sort=False, observed=True)
    }
    rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for saved in portfolio.sort_values("entry_time").itertuples(index=False):
        entry = pd.Timestamp(saved.entry_time)
        local_indexed = panel_groups[entry]
        local = local_indexed.reset_index(drop=True)
        current = weight_groups[entry]
        target = _target(local, feature_groups.get(entry))
        alt_symbols = [symbol for symbol in current if symbol != BTC]
        price = sum(
            current[symbol] * float(local_indexed.at[symbol, "price_return"])
            for symbol in alt_symbols
        ) + current[BTC] * float(local.iloc[0]["btc_return"])
        funding = -sum(
            current[symbol] * float(local_indexed.at[symbol, "future_funding"])
            for symbol in alt_symbols
        ) - current[BTC] * float(local.iloc[0]["btc_future_funding"])
        beta = sum(
            current[symbol] * float(local_indexed.at[symbol, "btc_beta"])
            for symbol in alt_symbols
        ) + current[BTC]
        rows.append(
            {
                "entry_time": entry,
                "rebalance_turnover": _turnover(previous, current),
                "target_tracking_l1": _turnover(current, target),
                "price_return": float(price),
                "funding_return": float(funding),
                "gross_return": float(price + funding),
                "gross_notional": float(sum(abs(value) for value in current.values())),
                "residual_btc_beta": float(beta),
                "terminal_notional": float(sum(abs(value) for value in current.values())),
            }
        )
        previous = current
    recomputed = pd.DataFrame(rows)
    recomputed["realized_turnover"] = recomputed["rebalance_turnover"]
    recomputed.loc[recomputed.index[-1], "realized_turnover"] += recomputed.iloc[-1][
        "terminal_notional"
    ]
    recomputed["primary_net_return"] = (
        recomputed["gross_return"]
        - cfg.one_way_cost * recomputed["realized_turnover"]
    )
    recomputed["stress_net_return"] = (
        recomputed["gross_return"]
        - cfg.stress_one_way_cost * recomputed["realized_turnover"]
    )
    columns = [
        "rebalance_turnover",
        "target_tracking_l1",
        "realized_turnover",
        "price_return",
        "funding_return",
        "gross_return",
        "gross_notional",
        "residual_btc_beta",
        "primary_net_return",
        "stress_net_return",
    ]
    merged = recomputed[["entry_time", *columns]].merge(
        portfolio[["entry_time", *columns]],
        on="entry_time",
        suffixes=("_audit", "_saved"),
        validate="one_to_one",
    )
    errors = pd.DataFrame(
        [
            {
                "field": column,
                "maximum_absolute_error": float(
                    (merged[f"{column}_audit"] - merged[f"{column}_saved"])
                    .abs()
                    .max()
                ),
            }
            for column in columns
        ]
    )
    return recomputed, errors


def _formula_errors(
    overlay: pd.DataFrame,
    cm2: pd.DataFrame,
    tg1: pd.DataFrame,
    comparison: pd.DataFrame,
    baseline: pd.DataFrame,
    saved_cm2: pd.DataFrame,
    reversed_control: pd.DataFrame,
    cfg: V223Config,
) -> pd.DataFrame:
    merged = cm2.merge(
        overlay[
            [
                "entry_time",
                "price_return",
                "funding_return",
                "primary_net_return",
                "stress_net_return",
            ]
        ],
        on="entry_time",
        suffixes=("_cm2", "_overlay"),
        validate="one_to_one",
    ).merge(
        tg1[
            [
                "entry_time",
                "price_basis_return",
                "funding_spread_return",
                "primary_net_return",
                "stress_net_return",
            ]
        ].rename(
            columns={
                "price_basis_return": "tg1_source_price",
                "funding_spread_return": "tg1_source_funding",
                "primary_net_return": "tg1_source_primary",
                "stress_net_return": "tg1_source_stress",
            }
        ),
        on="entry_time",
        validate="one_to_one",
    )
    expected = {
        "cm2_fss3_primary_component": merged["fss3_primary_return"]
        - merged["primary_net_return_overlay"],
        "cm2_tg1_primary_component": merged["tg1_primary_return"]
        - merged["tg1_source_primary"],
        "cm2_tg1_stress_component": merged["tg1_stress_return"]
        - merged["tg1_source_stress"],
        "cm2_price_formula": merged["price_return_cm2"]
        - (
            cfg.fss3_weight * merged["fss3_price_return"]
            + cfg.tg1_weight * merged["tg1_source_price"]
        ),
        "cm2_funding_formula": merged["funding_return_cm2"]
        - (
            cfg.fss3_weight * merged["fss3_funding_return"]
            + cfg.tg1_weight * merged["tg1_source_funding"]
        ),
        "cm2_primary_formula": merged["primary_net_return_cm2"]
        - (
            cfg.fss3_weight * merged["primary_net_return_overlay"]
            + cfg.tg1_weight * merged["tg1_source_primary"]
        ),
        "cm2_stress_formula": merged["stress_net_return_cm2"]
        - (
            cfg.fss3_weight * merged["stress_net_return_overlay"]
            + cfg.tg1_weight * merged["tg1_source_stress"]
        ),
    }
    comp = (
        comparison.merge(
            overlay[["entry_time", "primary_net_return"]].rename(
                columns={"primary_net_return": "audit_overlay_primary"}
            ),
            on="entry_time",
            validate="one_to_one",
        )
        .merge(
            baseline[["entry_time", "primary_net_return"]].rename(
                columns={"primary_net_return": "audit_baseline_primary"}
            ),
            on="entry_time",
            validate="one_to_one",
        )
        .merge(
            reversed_control[["entry_time", "primary_net_return"]].rename(
                columns={"primary_net_return": "audit_reversed_primary"}
            ),
            on="entry_time",
            validate="one_to_one",
        )
        .merge(
            cm2[["entry_time", "primary_net_return"]].rename(
                columns={"primary_net_return": "audit_overlay_cm2"}
            ),
            on="entry_time",
            validate="one_to_one",
        )
        .merge(
            saved_cm2[["entry_time", "primary_net_return"]].rename(
                columns={"primary_net_return": "audit_baseline_cm2"}
            ),
            on="entry_time",
            validate="one_to_one",
        )
    )
    expected.update(
        {
            "comparison_primary_increment": comp["incremental_primary_net_return"]
            - (comp["audit_overlay_primary"] - comp["audit_baseline_primary"]),
            "comparison_reversed_increment": comp[
                "reversed_incremental_primary_return"
            ]
            - (comp["audit_reversed_primary"] - comp["audit_baseline_primary"]),
            "comparison_cm2_increment": comp["incremental_cm2_primary_return"]
            - (comp["audit_overlay_cm2"] - comp["audit_baseline_cm2"]),
        }
    )
    return pd.DataFrame(
        [
            {"field": field, "maximum_absolute_error": float(values.abs().max())}
            for field, values in expected.items()
        ]
    )


def _summary_reconstruction(
    summary: pd.Series,
    overlay: pd.DataFrame,
    baseline: pd.DataFrame,
    cm2: pd.DataFrame,
    comparison: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V223Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = comparison[comparison["overlay_active"]].sort_values("entry_time")
    increment = active["incremental_primary_net_return"].to_numpy(dtype=float)
    draws = _moving_block_means(
        increment,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + 2),
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    periods = active.groupby("period", observed=True)[
        "incremental_primary_net_return"
    ].mean()
    months = active.groupby("month_start", observed=True)[
        "incremental_primary_net_return"
    ].sum()
    positive = months[months.gt(0)]
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    )
    lomo = min(
        float(
            active.loc[
                active["month_start"].ne(month), "incremental_primary_net_return"
            ].mean()
        )
        for month in months.index
    )
    cm2_active = cm2[cm2["overlay_active"]].sort_values("entry_time")
    baseline_cm2 = _utc(pd.read_parquet(cfg.cm2_path)).set_index("entry_time")
    base_active = baseline_cm2.loc[cm2_active["entry_time"]]
    observed = float(increment.mean())
    expected = {
        "full_path_overlay_fss3_primary_bp": overlay["primary_net_return"].mean()
        * 10_000,
        "full_path_baseline_fss3_primary_bp": baseline["primary_net_return"].mean()
        * 10_000,
        "full_path_fss3_increment_bp": comparison[
            "incremental_primary_net_return"
        ].mean()
        * 10_000,
        "active_fss3_primary_increment_bp": observed * 10_000,
        "active_fss3_stress_increment_bp": active[
            "incremental_stress_net_return"
        ].mean()
        * 10_000,
        "active_cm2_primary_increment_bp": active[
            "incremental_cm2_primary_return"
        ].mean()
        * 10_000,
        "active_cm2_stress_increment_bp": active[
            "incremental_cm2_stress_return"
        ].mean()
        * 10_000,
        "active_price_increment_bp": active["incremental_price_return"].mean()
        * 10_000,
        "active_funding_increment_bp": active["incremental_funding_return"].mean()
        * 10_000,
        "development_primary_increment_bp": periods.get("development", np.nan)
        * 10_000,
        "validation_primary_increment_bp": periods.get("validation", np.nan)
        * 10_000,
        "holdout_primary_increment_bp": periods.get("holdout", np.nan) * 10_000,
        "bootstrap_95_low_increment_bp": low * 10_000,
        "bootstrap_95_high_increment_bp": high * 10_000,
        "random_null_percentile": 100
        * nulls["mean_active_primary_increment"].le(observed).mean(),
        "reversed_active_primary_increment_bp": active[
            "reversed_incremental_primary_return"
        ].mean()
        * 10_000,
        "mean_overlay_turnover": overlay["realized_turnover"].mean(),
        "mean_baseline_turnover": baseline["realized_turnover"].mean(),
        "mean_turnover_increment": overlay["realized_turnover"].mean()
        - baseline["realized_turnover"].mean(),
        "max_capped_transition_turnover": overlay.loc[
            overlay["cap_applicable"], "rebalance_turnover"
        ].max(),
        "max_cap_breach": overlay["cap_breach"].max(),
        "max_abs_residual_btc_beta": overlay["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (overlay["gross_notional"] - 1).abs().max(),
        "positive_month_increment_concentration": concentration,
        "minimum_leave_one_month_out_increment_bp": lomo * 10_000,
        "overlay_cm2_active_drawdown_bp": _additive_max_drawdown(
            cm2_active["primary_net_return"]
        )
        * 10_000,
        "baseline_cm2_active_drawdown_bp": _additive_max_drawdown(
            base_active["primary_net_return"]
        )
        * 10_000,
    }
    expected["cm2_active_drawdown_worsening_bp"] = abs(
        expected["overlay_cm2_active_drawdown_bp"]
    ) - abs(expected["baseline_cm2_active_drawdown_bp"])
    errors = pd.DataFrame(
        [
            {
                "field": f"summary_{field}",
                "maximum_absolute_error": abs(float(summary[field]) - float(value)),
            }
            for field, value in expected.items()
        ]
    )
    gates = {
        "coverage_exact": (
            summary["path_weeks"] == 49
            and summary["active_weeks"] == 35
            and summary["active_months"] == 9
            and summary["development_active_weeks"] == 16
            and summary["validation_active_weeks"] == 11
            and summary["holdout_active_weeks"] == 8
        ),
        "baseline_reconstruction": summary["baseline_reconstruction_max_error"]
        <= cfg.tolerance,
        "positive_active_fss3_primary": summary["active_fss3_primary_increment_bp"]
        > 0,
        "positive_active_fss3_stress": summary["active_fss3_stress_increment_bp"]
        > 0,
        "positive_active_cm2_primary": summary["active_cm2_primary_increment_bp"] > 0,
        "positive_active_cm2_stress": summary["active_cm2_stress_increment_bp"] > 0,
        "positive_development": summary["development_primary_increment_bp"] > 0,
        "positive_validation": summary["validation_primary_increment_bp"] > 0,
        "positive_holdout": summary["holdout_primary_increment_bp"] > 0,
        "positive_bootstrap_lower": summary["bootstrap_95_low_increment_bp"] > 0,
        "random_percentile_95": summary["random_null_percentile"] >= 95,
        "beats_reversed": summary["active_fss3_primary_increment_bp"]
        > summary["reversed_active_primary_increment_bp"],
        "turnover_increment_within_010": summary["mean_turnover_increment"] <= 0.10,
        "cap_respected": summary["max_capped_transition_turnover"]
        <= cfg.turnover_cap + cfg.cap_tolerance,
        "no_cap_breach": summary["max_cap_breach"] <= cfg.cap_tolerance,
        "beta_exact": summary["max_abs_residual_btc_beta"] <= cfg.tolerance,
        "gross_exact": summary["max_gross_notional_drift"] <= cfg.tolerance,
        "month_concentration": summary["positive_month_increment_concentration"]
        <= 0.35,
        "positive_lomo": summary["minimum_leave_one_month_out_increment_bp"] > 0,
        "drawdown_worsening_within_200bp": summary[
            "cm2_active_drawdown_worsening_bp"
        ]
        <= 200,
    }
    gate_frame = pd.DataFrame({"gate": list(gates), "passed": list(gates.values())})
    return errors, gate_frame


def run_v223_audit(
    cfg: V223Config = V223Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _utc(pd.read_parquet(cfg.panel_path))
    features = _utc(pd.read_parquet(cfg.feature_path))
    portfolio = _utc(pd.read_parquet(cfg.v222_root / "weekly_overlay_fss3.parquet"))
    weights = _utc(pd.read_parquet(cfg.v222_root / "weekly_overlay_weights.parquet"))
    baseline = _utc(
        pd.read_parquet(cfg.v222_root / "weekly_zero_tilt_reconstruction.parquet")
    )
    reversed_control = _utc(
        pd.read_parquet(cfg.v222_root / "weekly_reversed_control.parquet")
    )
    cm2 = _utc(pd.read_parquet(cfg.v222_root / "weekly_overlay_cm2.parquet"))
    comparison = _utc(
        pd.read_parquet(cfg.v222_root / "weekly_increment_comparison.parquet")
    )
    tg1 = _utc(pd.read_parquet(cfg.tg1_path))
    saved_cm2 = _utc(pd.read_parquet(cfg.cm2_path))
    nulls = pd.read_csv(cfg.v222_root / "random_rank_nulls.csv")
    summary = pd.read_csv(cfg.v222_root / "summary.csv").iloc[0]
    reconstruction_saved = pd.read_csv(
        cfg.v222_root / "baseline_reconstruction_errors.csv"
    )
    metadata = json.loads((cfg.v222_root / "metadata.json").read_text(encoding="utf-8"))
    recomputed, pnl_errors = independently_recompute_overlay(
        panel, features, portfolio, weights, cfg
    )
    formula_errors = _formula_errors(
        portfolio,
        cm2,
        tg1,
        comparison,
        baseline,
        saved_cm2,
        reversed_control,
        cfg,
    )
    summary_errors, gates = _summary_reconstruction(
        summary, portfolio, baseline, cm2, comparison, nulls, cfg
    )
    maximum_errors = pd.concat(
        [pnl_errors, formula_errors, summary_errors], ignore_index=True
    )
    feature_checks = pd.read_csv(cfg.feature_checks_path)
    active = portfolio[portfolio["overlay_active"]]
    counts = active["period"].value_counts()
    feature_hash = _sha256(cfg.feature_path)
    all_gates = bool(gates["passed"].all())
    checks = {
        "v221_feature_audit_all_pass": bool(feature_checks["passed"].all()),
        "feature_hash_matches_preregistration": feature_hash == FEATURE_SHA256,
        "feature_hash_matches_v222_metadata": feature_hash
        == metadata["feature_sha256"],
        "feature_keys_unique": not features.duplicated(["entry_time", "symbol"]).any(),
        "feature_lag_is_causal": bool(
            (features["entry_time"] - features["sfi_feature_time"])
            .dt.total_seconds()
            .div(3600)
            .between(12, 36)
            .all()
        ),
        "portfolio_has_49_unique_weeks": len(portfolio) == 49
        and portfolio["entry_time"].is_unique,
        "weights_have_unique_week_symbol_keys": not weights.duplicated(
            ["entry_time", "symbol"]
        ).any(),
        "weight_and_portfolio_week_keys_match": set(weights["entry_time"])
        == set(portfolio["entry_time"]),
        "active_coverage_is_frozen_35_16_11_8": len(active) == 35
        and int(counts.get("development", 0)) == 16
        and int(counts.get("validation", 0)) == 11
        and int(counts.get("holdout", 0)) == 8,
        "independent_position_pnl_and_costs_exact": float(
            pnl_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "target_tracking_exact": float(
            pnl_errors.loc[
                pnl_errors["field"].eq("target_tracking_l1"),
                "maximum_absolute_error",
            ].iloc[0]
        )
        <= cfg.tolerance,
        "cm2_and_comparison_formulas_exact": float(
            formula_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "summary_statistics_exact": float(
            summary_errors["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "saved_zero_tilt_reconstruction_exact": float(
            reconstruction_saved["maximum_absolute_error"].max()
        )
        <= cfg.tolerance,
        "null_has_1000_unique_iterations": len(nulls) == 1000
        and nulls["iteration"].nunique() == 1000,
        "null_preserves_35_active_weeks": bool(nulls["active_weeks"].eq(35).all()),
        "cap_applicable_transitions_respect_070": bool(
            portfolio.loc[portfolio["cap_applicable"], "rebalance_turnover"]
            .le(cfg.turnover_cap + cfg.cap_tolerance)
            .all()
        ),
        "gross_and_beta_constraints_exact": bool(
            portfolio["gross_notional"].sub(1).abs().le(cfg.tolerance).all()
            and portfolio["residual_btc_beta"].abs().le(cfg.tolerance).all()
        ),
        "promotion_flag_matches_gate_conjunction": bool(summary["promote"])
        == all_gates,
        "failed_gates_force_rejection": (not all_gates) and (not bool(summary["promote"])),
        "metadata_has_no_promotion": metadata["promoted"] == [],
        "metadata_records_no_permission_change": metadata["permissions_changed"] == [],
        "findings_records_rejection": "Verdict: `reject_strategy_overlay`."
        in (cfg.v222_root.parent.parent / "docs/v222_sfi_fss3_overlay_findings_2026_07_17.md")
        .read_text(encoding="utf-8"),
    }
    audit_checks = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    hashes = pd.DataFrame(
        [
            {"artifact": str(cfg.panel_path), "sha256": _sha256(cfg.panel_path)},
            {"artifact": str(cfg.feature_path), "sha256": feature_hash},
            {
                "artifact": str(cfg.v222_root / "weekly_overlay_fss3.parquet"),
                "sha256": _sha256(cfg.v222_root / "weekly_overlay_fss3.parquet"),
            },
            {
                "artifact": str(cfg.v222_root / "random_rank_nulls.csv"),
                "sha256": _sha256(cfg.v222_root / "random_rank_nulls.csv"),
            },
        ]
    )
    return audit_checks, maximum_errors, gates, hashes


def write_v223_sfi_fss3_overlay_audit(
    cfg: V223Config = V223Config(),
) -> dict[str, Path]:
    checks, errors, gates, hashes = run_v223_audit(cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "checks": root / "audit_checks.csv",
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
                "failed_promotion_gates": failed_gates,
                "validated_verdict": "reject_strategy_overlay" if passed else "audit_failed",
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
                "# v22.3 SFI-on-FSS3 Independent Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Audit checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
                "",
                "## Failed promotion gates",
                "",
                ", ".join(f"`{gate}`" for gate in failed_gates),
                "",
                "The audit independently reconstructed position PnL, funding, turnover,",
                "costs, beta/gross constraints, CM2 arithmetic and inference summaries.",
                "It validates the v22.2 rejection; it does not promote the overlay.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
