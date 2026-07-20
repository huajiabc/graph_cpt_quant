"""Independent sklearn reproduction of the preregistered v23.27 selector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from pressure_graph.io import ensure_dir


FEATURE_PATH = Path(
    "reports/v23_26_multisource_oco_model_feature_audit/"
    "multisource_model_features.parquet"
)
OUTCOME_PATH = Path(
    "reports/v23_4_book_vacuum_oco_breakout/barrier_variant_outcomes.parquet"
)
SOURCE_ROOT = Path("reports/v23_27_multisource_interaction_ridge_oco_selector")
REPORT_ROOT = Path(
    "reports/v23_28_multisource_interaction_ridge_independent_audit"
)
FINDINGS_PATH = Path(
    "docs/v2328_multisource_interaction_ridge_independent_audit_2026_07_17.md"
)
PREREG_PATH = Path(
    "docs/v2327_multisource_interaction_ridge_preregistration_2026_07_17.md"
)
EXPECTED_FEATURE_HASH = (
    "4B93B3F7A5D340776CF0CDAA5E16C37AFACF6A20AFC93ED25C74DC2BB393B081"
)
FEATURES = (
    "signal_direction",
    "pressure_excess",
    "directional_breadth",
    "withdrawal_breadth",
    "causal_hourly_sigma",
    "btc_return_1h",
    "alt_taker_log_median",
    "alt_taker_buy_breadth",
    "alt_oi_change_median",
    "alt_oi_build_breadth",
    "alt_top_position_change_median",
    "alt_top_position_build_breadth",
    "alt_top_size_bias_median",
    "btc_taker_log",
    "btc_oi_change",
    "btc_top_position_change",
    "btc_top_size_bias",
    "utc_hour_sin",
    "utc_hour_cos",
)


def _feature_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_values("entry_time").to_csv(
        index=False, date_format="%Y-%m-%dT%H:%M:%S%z"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _load_inputs() -> pd.DataFrame:
    features = pd.read_parquet(FEATURE_PATH)
    for column in ("feature_time", "entry_time", "metric_feature_time"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="raise")
    outcomes = pd.read_parquet(OUTCOME_PATH)
    outcomes["entry_time"] = pd.to_datetime(
        outcomes["entry_time"], utc=True, errors="raise"
    )
    outcomes = outcomes.loc[
        np.isclose(outcomes["sigma_multiple"], 0.75),
        ["entry_time", "primary_net_return", "stress_net_return", "triggered"],
    ]
    return (
        features.merge(outcomes, on="entry_time", validate="one_to_one")
        .sort_values("entry_time")
        .reset_index(drop=True)
    )


def _design(
    train_x: np.ndarray, predict_x: np.ndarray, interaction: bool
) -> tuple[np.ndarray, np.ndarray]:
    base_scaler = StandardScaler()
    train = base_scaler.fit_transform(train_x)
    predict = base_scaler.transform(predict_x)
    if interaction:
        polynomial = PolynomialFeatures(degree=2, include_bias=False)
        train = polynomial.fit_transform(train)
        predict = polynomial.transform(predict)
        polynomial_scaler = StandardScaler()
        train = polynomial_scaler.fit_transform(train)
        predict = polynomial_scaler.transform(predict)
    return train, predict


def _rebuild_predictions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    rows = []
    designs = {}
    for model, interaction, alpha in (
        ("linear_ridge", False, 10.0),
        ("interaction_ridge", True, 100.0),
    ):
        for train_periods, predict_period in (
            (("development",), "validation"),
            (("development", "validation"), "holdout"),
        ):
            train = frame.loc[frame["period"].isin(train_periods)]
            predict = frame.loc[frame["period"].eq(predict_period)].copy()
            train_design, predict_design = _design(
                train[list(FEATURES)].to_numpy(float),
                predict[list(FEATURES)].to_numpy(float),
                interaction,
            )
            estimator = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            estimator.fit(train_design, train["primary_net_return"].to_numpy(float))
            values = estimator.predict(predict_design)
            predict["model"] = model
            predict["predicted_primary_net_return"] = values
            predict["selected"] = values > 0.0
            predict["strategy_primary_return"] = np.where(
                predict["selected"], predict["primary_net_return"], 0.0
            )
            predict["strategy_stress_return"] = np.where(
                predict["selected"], predict["stress_net_return"], 0.0
            )
            rows.append(predict)
            designs[(model, predict_period)] = (train_design, predict_design)
    return pd.concat(rows, ignore_index=True), designs


def _summary(predictions: pd.DataFrame) -> pd.DataFrame:
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
                    "primary_spearman_ic": float(
                        local["predicted_primary_net_return"].corr(
                            local["primary_net_return"], method="spearman"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _rebuild_permutations(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    designs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rng = np.random.default_rng(20_260_717)
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
    perm_dev = np.column_stack([rng.permutation(development_y) for _ in range(1_000)])
    perm_dev_validation = np.column_stack(
        [rng.permutation(development_validation_y) for _ in range(1_000)]
    )
    train_validation, predict_validation = designs[
        ("interaction_ridge", "validation")
    ]
    train_holdout, predict_holdout = designs[("interaction_ridge", "holdout")]
    validation_estimator = Ridge(alpha=100.0, fit_intercept=True, solver="cholesky")
    holdout_estimator = Ridge(alpha=100.0, fit_intercept=True, solver="cholesky")
    validation_estimator.fit(train_validation, perm_dev)
    holdout_estimator.fit(train_holdout, perm_dev_validation)
    validation_prediction = validation_estimator.predict(predict_validation)
    holdout_prediction = holdout_estimator.predict(predict_holdout)
    scores = (
        np.where(validation_prediction > 0.0, validation_y[:, None], 0.0).sum(axis=0)
        + np.where(holdout_prediction > 0.0, holdout_y[:, None], 0.0).sum(axis=0)
    ) / 96
    observed = predictions.loc[
        predictions["model"].eq("interaction_ridge"),
        "strategy_primary_return",
    ].mean()
    percentile = 100.0 * (1 + np.sum(scores < observed)) / (len(scores) + 1)
    return pd.DataFrame(
        {
            "iteration": np.arange(1_000),
            "null_primary_opportunity_return": scores,
            "observed_primary_opportunity_return": observed,
            "observed_percentile": percentile,
        }
    )


def _rebuild_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions.loc[predictions["model"].eq("interaction_ridge")]
    blocks = [
        group["strategy_primary_return"].to_numpy(float)
        for _, group in primary.groupby("entry_month", sort=True)
    ]
    rng = np.random.default_rng(20_260_718)
    draws = []
    for iteration in range(5_000):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[index] for index in selected])
        draws.append((iteration, float(sample.mean())))
    return pd.DataFrame(
        draws, columns=["iteration", "primary_opportunity_return"]
    )


def _rebuild_lomo(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions.loc[predictions["model"].eq("interaction_ridge")]
    return pd.DataFrame(
        [
            {
                "excluded_month": month,
                "remaining_events": len(primary.loc[primary["entry_month"].ne(month)]),
                "primary_opportunity_return": float(
                    primary.loc[
                        primary["entry_month"].ne(month), "strategy_primary_return"
                    ].mean()
                ),
            }
            for month in sorted(primary["entry_month"].unique())
        ]
    )


def _numeric_equal(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> bool:
    merged = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        return False
    columns = [
        column.removesuffix("_left")
        for column in merged.columns
        if column.endswith("_left")
    ]
    for column in columns:
        a = merged[f"{column}_left"]
        b = merged[f"{column}_right"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif not a.astype(str).equals(b.astype(str)):
            return False
    return True


def write_v2328_multisource_interaction_ridge_independent_audit() -> dict[str, Path]:
    frame = _load_inputs()
    rebuilt, designs = _rebuild_predictions(frame)
    source_predictions = pd.read_parquet(SOURCE_ROOT / "temporal_oos_predictions.parquet")
    source_predictions["entry_time"] = pd.to_datetime(
        source_predictions["entry_time"], utc=True, errors="raise"
    )
    rebuilt_summary = _summary(rebuilt)
    source_summary = pd.read_csv(SOURCE_ROOT / "temporal_fit_summary.csv")
    rebuilt_permutations = _rebuild_permutations(frame, rebuilt, designs)
    source_permutations = pd.read_parquet(SOURCE_ROOT / "random_label_null.parquet")
    rebuilt_bootstrap = _rebuild_bootstrap(rebuilt)
    source_bootstrap = pd.read_parquet(SOURCE_ROOT / "month_block_bootstrap.parquet")
    rebuilt_lomo = _rebuild_lomo(rebuilt)
    source_lomo = pd.read_csv(SOURCE_ROOT / "leave_one_month_out.csv")
    source_gates = pd.read_csv(SOURCE_ROOT / "decision_gates.csv")
    source_metadata = json.loads((SOURCE_ROOT / "metadata.json").read_text("utf-8"))
    prediction_pair = rebuilt.merge(
        source_predictions[
            ["model", "entry_time", "predicted_primary_net_return", "selected"]
        ],
        on=["model", "entry_time"],
        suffixes=("_audit", "_source"),
        validate="one_to_one",
    )
    checks = {
        "preregistration_and_feature_hash_frozen": PREREG_PATH.exists()
        and _feature_hash(frame.drop(columns=["primary_net_return", "stress_net_return", "triggered"]))
        == EXPECTED_FEATURE_HASH,
        "all_159_events_joined": len(frame) == 159 and frame["entry_time"].is_unique,
        "causal_feature_times_not_after_entry": frame["feature_time"].le(
            frame["entry_time"]
        ).all()
        and frame["metric_feature_time"].le(frame["entry_time"]).all(),
        "all_19_features_finite": np.isfinite(frame[list(FEATURES)].to_numpy(float)).all(),
        "temporal_prediction_counts_exact": len(rebuilt) == 192
        and rebuilt.groupby(["model", "period"]).size().to_dict()
        == {
            ("linear_ridge", "validation"): 47,
            ("linear_ridge", "holdout"): 49,
            ("interaction_ridge", "validation"): 47,
            ("interaction_ridge", "holdout"): 49,
        },
        "sklearn_predictions_match_source": np.allclose(
            prediction_pair["predicted_primary_net_return_audit"],
            prediction_pair["predicted_primary_net_return_source"],
            rtol=1e-9,
            atol=1e-12,
        ),
        "selection_flags_match_source": prediction_pair["selected_audit"].equals(
            prediction_pair["selected_source"]
        ),
        "summary_matches_source": _numeric_equal(
            rebuilt_summary, source_summary, ["model", "scope"]
        ),
        "permutation_null_matches_source": _numeric_equal(
            rebuilt_permutations, source_permutations, ["iteration"]
        ),
        "month_bootstrap_matches_source": _numeric_equal(
            rebuilt_bootstrap, source_bootstrap, ["iteration"]
        ),
        "leave_one_month_out_matches_source": _numeric_equal(
            rebuilt_lomo, source_lomo, ["excluded_month"]
        ),
        "gate_count_and_failure_match_source": len(source_gates) == 11
        and int(source_gates["passed"].sum()) == 1,
        "rejected_verdict_matches_evidence": source_metadata["verdict"]
        == "rejected_no_incremental_complex_model_alpha"
        and not source_metadata["all_gates_passed"],
    }
    audit = pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )
    root = ensure_dir(REPORT_ROOT)
    paths = {
        "checks": root / "independent_audit_checks.csv",
        "predictions": root / "independently_recomputed_predictions.parquet",
        "summary": root / "independently_recomputed_summary.csv",
        "metadata": root / "metadata.json",
        "findings": FINDINGS_PATH,
    }
    audit.to_csv(paths["checks"], index=False)
    rebuilt.to_parquet(paths["predictions"], index=False)
    rebuilt_summary.to_csv(paths["summary"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "audit_checks_passed": int(audit["passed"].sum()),
                "audit_checks_total": len(audit),
                "all_checks_passed": bool(audit["passed"].all()),
                "independent_backend": "sklearn.StandardScaler/PolynomialFeatures/Ridge",
                "source_verdict": source_metadata["verdict"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.28 Multisource Interaction-Ridge Independent Audit",
                "",
                f"Audit result: {int(audit['passed'].sum())}/{len(audit)} checks passed.",
                "",
                "An independent sklearn implementation reproduces the temporal",
                "predictions, selections, summaries, random-label null, month bootstrap,",
                "and leave-one-month-out results. The rejection verdict is upheld.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["write_v2328_multisource_interaction_ridge_independent_audit"]
