"""Preregistered temporal ridge selectors for the fixed 0.75-sigma OCO payoff."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2326_multisource_oco_model_feature_audit import (
    MODEL_FEATURES,
    feature_hash_v2326,
)


FEATURE_PATH = Path(
    "reports/v23_26_multisource_oco_model_feature_audit/"
    "multisource_model_features.parquet"
)
OUTCOME_PATH = Path(
    "reports/v23_4_book_vacuum_oco_breakout/barrier_variant_outcomes.parquet"
)
REPORT_ROOT = Path("reports/v23_27_multisource_interaction_ridge_oco_selector")
FINDINGS_PATH = Path(
    "docs/v2327_multisource_interaction_ridge_oco_selector_2026_07_17.md"
)
PREREG_PATH = Path(
    "docs/v2327_multisource_interaction_ridge_preregistration_2026_07_17.md"
)
CANDIDATE = "MSM1_MULTISOURCE_INTERACTION_RIDGE_OCO_SELECTOR"
EXPECTED_FEATURE_HASH = (
    "4B93B3F7A5D340776CF0CDAA5E16C37AFACF6A20AFC93ED25C74DC2BB393B081"
)
FEATURE_GROUPS = {
    "book_pressure": (
        "signal_direction",
        "pressure_excess",
        "directional_breadth",
        "withdrawal_breadth",
    ),
    "btc_price_vol": ("causal_hourly_sigma", "btc_return_1h"),
    "alt_derivatives": (
        "alt_taker_log_median",
        "alt_taker_buy_breadth",
        "alt_oi_change_median",
        "alt_oi_build_breadth",
        "alt_top_position_change_median",
        "alt_top_position_build_breadth",
        "alt_top_size_bias_median",
    ),
    "btc_derivatives": (
        "btc_taker_log",
        "btc_oi_change",
        "btc_top_position_change",
        "btc_top_size_bias",
    ),
    "clock": ("utc_hour_sin", "utc_hour_cos"),
}


@dataclass(frozen=True)
class V2327Config:
    feature_path: Path = FEATURE_PATH
    outcome_path: Path = OUTCOME_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    sigma_multiple: float = 0.75
    linear_alpha: float = 10.0
    interaction_alpha: float = 100.0
    prediction_cutoff: float = 0.0
    minimum_period_trades: int = 8
    minimum_active_months: int = 5
    minimum_positive_month_fraction: float = 0.60
    permutation_iterations: int = 1_000
    bootstrap_iterations: int = 5_000
    seed: int = 20_260_717


def _safe_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[~np.isfinite(scale) | np.isclose(scale, 0.0)] = 1.0
    return mean, scale


def _quadratic_terms(values: np.ndarray) -> np.ndarray:
    columns = [values]
    for left in range(values.shape[1]):
        for right in range(left, values.shape[1]):
            columns.append((values[:, left] * values[:, right])[:, None])
    return np.hstack(columns)


def _prepare_designs(
    train_x: np.ndarray,
    predict_x: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = _safe_scale(train_x)
    train_base = (train_x - mean) / scale
    predict_base = (predict_x - mean) / scale
    if model == "linear_ridge":
        return train_base, predict_base
    if model != "interaction_ridge":
        raise ValueError(f"unknown model: {model}")
    train_poly = _quadratic_terms(train_base)
    predict_poly = _quadratic_terms(predict_base)
    poly_mean, poly_scale = _safe_scale(train_poly)
    return (
        (train_poly - poly_mean) / poly_scale,
        (predict_poly - poly_mean) / poly_scale,
    )


def _ridge_prediction_operator(
    train_design: np.ndarray,
    predict_design: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return K where prediction = mean(y) + K @ (y - mean(y))."""
    train_mean = train_design.mean(axis=0)
    centered = train_design - train_mean
    predict_centered = predict_design - train_mean
    gram = centered @ centered.T
    regularized = gram + alpha * np.eye(len(centered))
    return predict_centered @ centered.T @ np.linalg.solve(
        regularized, np.eye(len(centered))
    )


def _predict_from_operator(operator: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if targets.ndim == 1:
        mean = targets.mean()
        return mean + operator @ (targets - mean)
    mean = targets.mean(axis=0, keepdims=True)
    return mean + operator @ (targets - mean)


def load_v2327_inputs(
    cfg: V2327Config = V2327Config(),
) -> tuple[pd.DataFrame, str]:
    features = pd.read_parquet(cfg.feature_path)
    for column in ("feature_time", "entry_time", "metric_feature_time"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="raise")
    feature_hash = feature_hash_v2326(features)
    if feature_hash != EXPECTED_FEATURE_HASH:
        raise ValueError(f"feature hash changed: {feature_hash}")
    outcomes = pd.read_parquet(cfg.outcome_path)
    outcomes["entry_time"] = pd.to_datetime(
        outcomes["entry_time"], utc=True, errors="raise"
    )
    outcomes = outcomes.loc[
        np.isclose(outcomes["sigma_multiple"], cfg.sigma_multiple),
        ["entry_time", "primary_net_return", "stress_net_return", "triggered"],
    ]
    if len(outcomes) != 159 or not outcomes["entry_time"].is_unique:
        raise ValueError("fixed 0.75-sigma outcome set is not exactly 159 unique events")
    merged = features.merge(outcomes, on="entry_time", how="inner", validate="one_to_one")
    if len(merged) != 159:
        raise ValueError("feature/outcome join did not retain all 159 events")
    return merged.sort_values("entry_time").reset_index(drop=True), feature_hash


def _period_fit(
    frame: pd.DataFrame,
    train_periods: tuple[str, ...],
    predict_period: str,
    model: str,
    alpha: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    train = frame.loc[frame["period"].isin(train_periods)]
    predict = frame.loc[frame["period"].eq(predict_period)].copy()
    train_x = train[list(MODEL_FEATURES)].to_numpy(float)
    predict_x = predict[list(MODEL_FEATURES)].to_numpy(float)
    train_design, predict_design = _prepare_designs(train_x, predict_x, model)
    operator = _ridge_prediction_operator(train_design, predict_design, alpha)
    prediction = _predict_from_operator(
        operator, train["primary_net_return"].to_numpy(float)
    )
    predict["model"] = model
    predict["predicted_primary_net_return"] = prediction
    predict["selected"] = prediction > 0.0
    predict["strategy_primary_return"] = np.where(
        predict["selected"], predict["primary_net_return"], 0.0
    )
    predict["strategy_stress_return"] = np.where(
        predict["selected"], predict["stress_net_return"], 0.0
    )
    return predict, operator


def build_v2327_predictions(
    frame: pd.DataFrame,
    cfg: V2327Config = V2327Config(),
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows = []
    operators = {}
    for model, alpha in (
        ("linear_ridge", cfg.linear_alpha),
        ("interaction_ridge", cfg.interaction_alpha),
    ):
        validation, validation_operator = _period_fit(
            frame, ("development",), "validation", model, alpha
        )
        holdout, holdout_operator = _period_fit(
            frame, ("development", "validation"), "holdout", model, alpha
        )
        rows.extend([validation, holdout])
        operators[(model, "validation")] = validation_operator
        operators[(model, "holdout")] = holdout_operator
    return pd.concat(rows, ignore_index=True), operators


def _spearman(frame: pd.DataFrame) -> float:
    return float(
        frame["predicted_primary_net_return"].corr(
            frame["primary_net_return"], method="spearman"
        )
    )


def summarize_v2327(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("linear_ridge", "interaction_ridge"):
        model_frame = predictions.loc[predictions["model"].eq(model)]
        for scope in ("validation", "holdout", "oos"):
            local = (
                model_frame
                if scope == "oos"
                else model_frame.loc[model_frame["period"].eq(scope)]
            )
            selected = local.loc[local["selected"]]
            rows.append(
                {
                    "model": model,
                    "scope": scope,
                    "events": len(local),
                    "selected_trades": len(selected),
                    "selection_rate": float(local["selected"].mean()),
                    "primary_selected_expectancy_bp": float(
                        selected["primary_net_return"].mean() * 10_000
                    ),
                    "stress_selected_expectancy_bp": float(
                        selected["stress_net_return"].mean() * 10_000
                    ),
                    "primary_opportunity_return_bp": float(
                        local["strategy_primary_return"].mean() * 10_000
                    ),
                    "stress_opportunity_return_bp": float(
                        local["strategy_stress_return"].mean() * 10_000
                    ),
                    "unfiltered_primary_return_bp": float(
                        local["primary_net_return"].mean() * 10_000
                    ),
                    "primary_spearman_ic": _spearman(local),
                }
            )
    return pd.DataFrame(rows)


def month_diagnostics_v2327(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions.loc[predictions["model"].eq("interaction_ridge")]
    return (
        primary.groupby("entry_month", sort=True)
        .agg(
            events=("entry_time", "size"),
            selected_trades=("selected", "sum"),
            primary_return=("strategy_primary_return", "mean"),
            stress_return=("strategy_stress_return", "mean"),
        )
        .reset_index()
        .assign(
            primary_return_bp=lambda x: x["primary_return"] * 10_000,
            stress_return_bp=lambda x: x["stress_return"] * 10_000,
        )
    )


def posthoc_source_ablation_v2327(
    frame: pd.DataFrame,
    cfg: V2327Config = V2327Config(),
) -> pd.DataFrame:
    """Outcome-seen source diagnostics; these variants cannot be promoted."""
    rows = []
    all_features = tuple(MODEL_FEATURES)
    variants = {}
    for group, columns in FEATURE_GROUPS.items():
        variants[f"drop_{group}"] = tuple(
            column for column in all_features if column not in columns
        )
        variants[f"only_{group}"] = columns
    for variant, columns in variants.items():
        period_predictions = []
        for train_periods, predict_period in (
            (("development",), "validation"),
            (("development", "validation"), "holdout"),
        ):
            train = frame.loc[frame["period"].isin(train_periods)]
            predict = frame.loc[frame["period"].eq(predict_period)].copy()
            train_design, predict_design = _prepare_designs(
                train[list(columns)].to_numpy(float),
                predict[list(columns)].to_numpy(float),
                "interaction_ridge",
            )
            operator = _ridge_prediction_operator(
                train_design, predict_design, cfg.interaction_alpha
            )
            prediction = _predict_from_operator(
                operator, train["primary_net_return"].to_numpy(float)
            )
            predict["predicted_primary_net_return"] = prediction
            predict["selected"] = prediction > cfg.prediction_cutoff
            predict["strategy_primary_return"] = np.where(
                predict["selected"], predict["primary_net_return"], 0.0
            )
            predict["strategy_stress_return"] = np.where(
                predict["selected"], predict["stress_net_return"], 0.0
            )
            period_predictions.append(predict)
        combined = pd.concat(period_predictions, ignore_index=True)
        for scope in ("validation", "holdout", "oos"):
            local = (
                combined
                if scope == "oos"
                else combined.loc[combined["period"].eq(scope)]
            )
            selected = local.loc[local["selected"]]
            rows.append(
                {
                    "variant": variant,
                    "feature_count": len(columns),
                    "scope": scope,
                    "selected_trades": len(selected),
                    "primary_selected_expectancy_bp": float(
                        selected["primary_net_return"].mean() * 10_000
                    ),
                    "primary_opportunity_return_bp": float(
                        local["strategy_primary_return"].mean() * 10_000
                    ),
                    "stress_opportunity_return_bp": float(
                        local["strategy_stress_return"].mean() * 10_000
                    ),
                    "primary_spearman_ic": _spearman(local),
                    "promotion_eligible": False,
                }
            )
    return pd.DataFrame(rows)


def permutation_control_v2327(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    operators: dict[tuple[str, str], np.ndarray],
    cfg: V2327Config = V2327Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    development_y = frame.loc[
        frame["period"].eq("development"), "primary_net_return"
    ].to_numpy(float)
    development_validation_y = frame.loc[
        frame["period"].isin(("development", "validation")),
        "primary_net_return",
    ].to_numpy(float)
    validation_y = frame.loc[
        frame["period"].eq("validation"), "primary_net_return"
    ].to_numpy(float)
    holdout_y = frame.loc[
        frame["period"].eq("holdout"), "primary_net_return"
    ].to_numpy(float)
    perm_dev = np.column_stack(
        [rng.permutation(development_y) for _ in range(cfg.permutation_iterations)]
    )
    perm_dev_validation = np.column_stack(
        [
            rng.permutation(development_validation_y)
            for _ in range(cfg.permutation_iterations)
        ]
    )
    validation_prediction = _predict_from_operator(
        operators[("interaction_ridge", "validation")], perm_dev
    )
    holdout_prediction = _predict_from_operator(
        operators[("interaction_ridge", "holdout")], perm_dev_validation
    )
    validation_strategy = np.where(
        validation_prediction > cfg.prediction_cutoff, validation_y[:, None], 0.0
    )
    holdout_strategy = np.where(
        holdout_prediction > cfg.prediction_cutoff, holdout_y[:, None], 0.0
    )
    scores = (
        validation_strategy.sum(axis=0) + holdout_strategy.sum(axis=0)
    ) / (len(validation_y) + len(holdout_y))
    observed = predictions.loc[
        predictions["model"].eq("interaction_ridge"),
        "strategy_primary_return",
    ].mean()
    percentile = 100.0 * (1 + np.sum(scores < observed)) / (len(scores) + 1)
    return pd.DataFrame(
        {
            "iteration": np.arange(cfg.permutation_iterations),
            "null_primary_opportunity_return": scores,
            "observed_primary_opportunity_return": observed,
            "observed_percentile": percentile,
        }
    )


def bootstrap_v2327(
    predictions: pd.DataFrame,
    cfg: V2327Config = V2327Config(),
) -> pd.DataFrame:
    primary = predictions.loc[predictions["model"].eq("interaction_ridge")]
    blocks = [
        group["strategy_primary_return"].to_numpy(float)
        for _, group in primary.groupby("entry_month", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed + 1)
    draws = []
    for iteration in range(cfg.bootstrap_iterations):
        selection = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[index] for index in selection])
        draws.append(
            {
                "iteration": iteration,
                "primary_opportunity_return": float(sample.mean()),
            }
        )
    return pd.DataFrame(draws)


def lomo_v2327(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions.loc[predictions["model"].eq("interaction_ridge")]
    rows = []
    for month in sorted(primary["entry_month"].unique()):
        local = primary.loc[primary["entry_month"].ne(month)]
        rows.append(
            {
                "excluded_month": month,
                "remaining_events": len(local),
                "primary_opportunity_return": float(
                    local["strategy_primary_return"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def gates_v2327(
    summary: pd.DataFrame,
    months: pd.DataFrame,
    permutations: pd.DataFrame,
    bootstrap: pd.DataFrame,
    lomo: pd.DataFrame,
    cfg: V2327Config = V2327Config(),
) -> pd.DataFrame:
    def value(model: str, scope: str, column: str) -> float:
        row = summary.loc[summary["model"].eq(model) & summary["scope"].eq(scope)]
        return float(row[column].iloc[0])

    comparisons = []
    for scope in ("validation", "holdout", "oos"):
        comparisons.append(
            value("interaction_ridge", scope, "primary_opportunity_return_bp")
            > value("linear_ridge", scope, "primary_opportunity_return_bp")
        )
        comparisons.append(
            value("interaction_ridge", scope, "primary_opportunity_return_bp")
            > value("interaction_ridge", scope, "unfiltered_primary_return_bp")
        )
    active = months.loc[months["selected_trades"].gt(0)]
    checks = {
        "minimum_8_trades_each_period": all(
            value("interaction_ridge", scope, "selected_trades")
            >= cfg.minimum_period_trades
            for scope in ("validation", "holdout")
        ),
        "positive_primary_selected_each_period": all(
            value("interaction_ridge", scope, "primary_selected_expectancy_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "positive_stress_selected_each_period": all(
            value("interaction_ridge", scope, "stress_selected_expectancy_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "positive_primary_opportunity_each_period": all(
            value("interaction_ridge", scope, "primary_opportunity_return_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "positive_stress_opportunity_each_period": all(
            value("interaction_ridge", scope, "stress_opportunity_return_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "positive_spearman_each_period": all(
            value("interaction_ridge", scope, "primary_spearman_ic") > 0
            for scope in ("validation", "holdout")
        ),
        "beats_linear_and_unfiltered_all_scopes": all(comparisons),
        "month_activity_and_positive_fraction": len(active)
        >= cfg.minimum_active_months
        and float(active["primary_return"].gt(0).mean())
        >= cfg.minimum_positive_month_fraction,
        "all_leave_one_month_out_positive": lomo[
            "primary_opportunity_return"
        ].gt(0).all(),
        "month_bootstrap_q05_positive": float(
            bootstrap["primary_opportunity_return"].quantile(0.05)
        )
        > 0,
        "random_label_percentile_at_least_95": float(
            permutations["observed_percentile"].iloc[0]
        )
        >= 95.0,
    }
    return pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in checks.items()]
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_v2327_multisource_interaction_ridge_oco_selector(
    cfg: V2327Config = V2327Config(),
) -> dict[str, Path]:
    frame, feature_hash = load_v2327_inputs(cfg)
    predictions, operators = build_v2327_predictions(frame, cfg)
    summary = summarize_v2327(predictions)
    months = month_diagnostics_v2327(predictions)
    ablation = posthoc_source_ablation_v2327(frame, cfg)
    permutations = permutation_control_v2327(frame, predictions, operators, cfg)
    bootstrap = bootstrap_v2327(predictions, cfg)
    lomo = lomo_v2327(predictions)
    gates = gates_v2327(summary, months, permutations, bootstrap, lomo, cfg)
    passed = bool(gates["passed"].all())
    verdict = (
        "research_candidate_requires_isolated_forward_shadow"
        if passed
        else "rejected_no_incremental_complex_model_alpha"
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "predictions": root / "temporal_oos_predictions.parquet",
        "summary": root / "temporal_fit_summary.csv",
        "months": root / "oos_month_diagnostics.csv",
        "posthoc_ablation": root / "posthoc_source_ablation.csv",
        "permutations": root / "random_label_null.parquet",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "lomo": root / "leave_one_month_out.csv",
        "gates": root / "decision_gates.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    predictions.to_parquet(paths["predictions"], index=False)
    summary.to_csv(paths["summary"], index=False)
    months.to_csv(paths["months"], index=False)
    ablation.to_csv(paths["posthoc_ablation"], index=False)
    permutations.to_parquet(paths["permutations"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    lomo.to_csv(paths["lomo"], index=False)
    gates.to_csv(paths["gates"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "verdict": verdict,
                "all_gates_passed": passed,
                "feature_hash": feature_hash,
                "outcome_file_hash": _file_hash(cfg.outcome_path),
                "linear_design_columns": len(MODEL_FEATURES),
                "interaction_design_columns": len(MODEL_FEATURES)
                + len(MODEL_FEATURES) * (len(MODEL_FEATURES) + 1) // 2,
                "config": {
                    **asdict(cfg),
                    "feature_path": str(cfg.feature_path),
                    "outcome_path": str(cfg.outcome_path),
                    "report_root": str(cfg.report_root),
                    "findings_path": str(cfg.findings_path),
                    "prereg_path": str(cfg.prereg_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    interaction = summary.loc[summary["model"].eq("interaction_ridge")]
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.27 Multisource Interaction-Ridge OCO Selector",
                "",
                f"Verdict: `{verdict}`.",
                "",
                interaction.to_markdown(index=False, floatfmt=".4f"),
                "",
                f"Passed gates: {int(gates['passed'].sum())}/{len(gates)}.",
                f"Random-label percentile: {permutations['observed_percentile'].iloc[0]:.2f}.",
                "Month-bootstrap q05 (bp/opportunity): "
                f"{bootstrap['primary_opportunity_return'].quantile(0.05) * 10_000:.4f}.",
                "",
                "The model uses only causal v23.26 features and fixed temporal fits.",
                "Source ablations are outcome-seen diagnostics and are not promotion-eligible.",
                "No PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2327Config",
    "_predict_from_operator",
    "_prepare_designs",
    "_ridge_prediction_operator",
    "build_v2327_predictions",
    "gates_v2327",
    "load_v2327_inputs",
    "posthoc_source_ablation_v2327",
    "summarize_v2327",
    "write_v2327_multisource_interaction_ridge_oco_selector",
]
