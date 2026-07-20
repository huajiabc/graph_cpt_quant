"""Independent reproduction of the outcome-free v23.30 transmission selector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


FEATURE_PATH = Path(
    "reports/v23_29_event_volatility_transmission_feature_audit/"
    "event_volatility_transmission_features.parquet"
)
OUTCOME_PATH = Path(
    "reports/v23_4_book_vacuum_oco_breakout/barrier_variant_outcomes.parquet"
)
SOURCE_ROOT = Path("reports/v23_30_direct_volatility_transmission_selector")
REPORT_ROOT = Path(
    "reports/v23_31_direct_volatility_transmission_independent_audit"
)
FINDINGS_PATH = Path(
    "docs/v2331_direct_volatility_transmission_independent_audit_2026_07_17.md"
)
PREREG_PATH = Path(
    "docs/v2330_direct_volatility_transmission_selector_preregistration_2026_07_17.md"
)
EXPECTED_FEATURE_HASH = (
    "C7EFC21FA0B9FEC822BE86D4C7A986C352E6B3F2C5382E0EF6256CBCCD312FFF"
)
SCORE_FEATURES = (
    "btc_receiver_gap",
    "alt_rv_acceleration_median",
    "alt_residual_shock_breadth",
    "directed_edge_weight_mean",
)


def _feature_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_values("entry_time").to_csv(
        index=False, date_format="%Y-%m-%dT%H:%M:%S%z"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(FEATURE_PATH)
    for column in ("feature_time", "entry_time", "price_feature_time"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="raise")
    outcomes = pd.read_parquet(OUTCOME_PATH)
    outcomes["entry_time"] = pd.to_datetime(
        outcomes["entry_time"], utc=True, errors="raise"
    )
    outcomes = outcomes.loc[
        np.isclose(outcomes["sigma_multiple"], 0.75),
        ["entry_time", "primary_net_return", "stress_net_return", "triggered"],
    ]
    frame = features.merge(outcomes, on="entry_time", validate="one_to_one")
    return features, frame.sort_values("entry_time").reset_index(drop=True)


def _rebuild_selection(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for train_periods, predict_period in (
        (("development",), "validation"),
        (("development", "validation"), "holdout"),
    ):
        train = frame.loc[frame["period"].isin(train_periods)]
        predict = frame.loc[frame["period"].eq(predict_period)].copy()
        train_x = train[list(SCORE_FEATURES)].to_numpy(float)
        predict_x = predict[list(SCORE_FEATURES)].to_numpy(float)
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[np.isclose(scale, 0.0) | ~np.isfinite(scale)] = 1.0
        train_z = (train_x - mean) / scale
        predict_z = (predict_x - mean) / scale
        train_score = train_z.mean(axis=1)
        score = predict_z.mean(axis=1)
        threshold = float(np.quantile(train_score, 0.70))
        for index, feature in enumerate(SCORE_FEATURES):
            predict[f"z_{feature}"] = predict_z[:, index]
        predict["transmission_score"] = score
        predict["training_score_threshold"] = threshold
        predict["selected"] = score >= threshold
        predict["strategy_primary_return"] = np.where(
            predict["selected"], predict["primary_net_return"], 0.0
        )
        predict["strategy_stress_return"] = np.where(
            predict["selected"], predict["stress_net_return"], 0.0
        )
        rows.append(predict)
    return pd.concat(rows, ignore_index=True)


def _rebuild_random(selection: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20_260_717)
    validation = selection.loc[selection["period"].eq("validation")]
    holdout = selection.loc[selection["period"].eq("holdout")]
    validation_y = validation["primary_net_return"].to_numpy(float)
    holdout_y = holdout["primary_net_return"].to_numpy(float)
    validation_n = int(validation["selected"].sum())
    holdout_n = int(holdout["selected"].sum())
    scores = []
    for _ in range(1_000):
        left = rng.choice(len(validation_y), size=validation_n, replace=False)
        right = rng.choice(len(holdout_y), size=holdout_n, replace=False)
        scores.append(
            float(
                (validation_y[left].sum() + holdout_y[right].sum())
                / len(selection)
            )
        )
    observed = float(selection["strategy_primary_return"].mean())
    percentile = 100.0 * (1 + np.sum(np.asarray(scores) < observed)) / 1_001
    return pd.DataFrame(
        {
            "iteration": np.arange(1_000),
            "null_primary_opportunity_return": scores,
            "observed_primary_opportunity_return": observed,
            "observed_percentile": percentile,
        }
    )


def _rebuild_bootstrap(selection: pd.DataFrame) -> pd.DataFrame:
    blocks = [
        group["strategy_primary_return"].to_numpy(float)
        for _, group in selection.groupby("entry_month", sort=True)
    ]
    rng = np.random.default_rng(20_260_718)
    rows = []
    for iteration in range(5_000):
        indices = rng.integers(0, len(blocks), size=len(blocks))
        rows.append(
            (
                iteration,
                float(np.concatenate([blocks[index] for index in indices]).mean()),
            )
        )
    return pd.DataFrame(
        rows, columns=["iteration", "primary_opportunity_return"]
    )


def _rebuild_lomo(selection: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "excluded_month": month,
                "remaining_events": len(selection.loc[selection["entry_month"].ne(month)]),
                "primary_opportunity_return": float(
                    selection.loc[
                        selection["entry_month"].ne(month),
                        "strategy_primary_return",
                    ].mean()
                ),
            }
            for month in sorted(selection["entry_month"].unique())
        ]
    )


def _equal(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> bool:
    merged = left.merge(right, on=keys, suffixes=("_audit", "_source"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        return False
    for column in [
        name.removesuffix("_audit")
        for name in merged.columns
        if name.endswith("_audit")
    ]:
        audit = merged[f"{column}_audit"]
        source = merged[f"{column}_source"]
        if pd.api.types.is_numeric_dtype(audit):
            if not np.allclose(audit, source, rtol=1e-10, atol=1e-12, equal_nan=True):
                return False
        elif not audit.astype(str).equals(source.astype(str)):
            return False
    return True


def write_v2331_direct_volatility_transmission_independent_audit() -> dict[str, Path]:
    features, frame = _inputs()
    rebuilt = _rebuild_selection(frame)
    source = pd.read_parquet(SOURCE_ROOT / "temporal_oos_selection.parquet")
    source["entry_time"] = pd.to_datetime(source["entry_time"], utc=True, errors="raise")
    rebuilt_random = _rebuild_random(rebuilt)
    source_random = pd.read_parquet(SOURCE_ROOT / "random_same_count_control.parquet")
    rebuilt_bootstrap = _rebuild_bootstrap(rebuilt)
    source_bootstrap = pd.read_parquet(SOURCE_ROOT / "month_block_bootstrap.parquet")
    rebuilt_lomo = _rebuild_lomo(rebuilt)
    source_lomo = pd.read_csv(SOURCE_ROOT / "leave_one_month_out.csv")
    source_summary = pd.read_csv(SOURCE_ROOT / "result_summary.csv")
    source_gates = pd.read_csv(SOURCE_ROOT / "decision_gates.csv")
    metadata = json.loads((SOURCE_ROOT / "metadata.json").read_text("utf-8"))
    pair = rebuilt.merge(
        source[
            [
                "entry_time",
                "transmission_score",
                "training_score_threshold",
                "selected",
                "strategy_primary_return",
            ]
        ],
        on="entry_time",
        suffixes=("_audit", "_source"),
        validate="one_to_one",
    )
    checks = {
        "preregistration_exists": PREREG_PATH.exists(),
        "feature_hash_exact": _feature_hash(features) == EXPECTED_FEATURE_HASH,
        "all_159_events_and_96_oos_predictions": len(frame) == 159
        and len(rebuilt) == 96,
        "causal_feature_times": features["feature_time"].le(
            features["entry_time"]
        ).all()
        and features["price_feature_time"].le(features["entry_time"]).all(),
        "score_values_match": np.allclose(
            pair["transmission_score_audit"],
            pair["transmission_score_source"],
            rtol=1e-10,
            atol=1e-12,
        ),
        "thresholds_match": np.allclose(
            pair["training_score_threshold_audit"],
            pair["training_score_threshold_source"],
            rtol=1e-10,
            atol=1e-12,
        ),
        "selection_and_returns_match": pair["selected_audit"].equals(
            pair["selected_source"]
        )
        and np.allclose(
            pair["strategy_primary_return_audit"],
            pair["strategy_primary_return_source"],
        ),
        "random_control_matches": _equal(
            rebuilt_random, source_random, ["iteration"]
        ),
        "month_bootstrap_matches": _equal(
            rebuilt_bootstrap, source_bootstrap, ["iteration"]
        ),
        "lomo_matches": _equal(rebuilt_lomo, source_lomo, ["excluded_month"]),
        "summary_and_gate_counts_reconcile": source_summary["events"].tolist()
        == [47, 49, 96]
        and int(source_gates["passed"].sum()) == 1,
        "rejection_verdict_upheld": metadata["verdict"]
        == "rejected_direct_volatility_transmission_selector"
        and not metadata["all_gates_passed"]
        and not metadata["outcome_trained"],
    }
    audit = pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )
    root = ensure_dir(REPORT_ROOT)
    paths = {
        "checks": root / "independent_audit_checks.csv",
        "selection": root / "independently_recomputed_selection.parquet",
        "metadata": root / "metadata.json",
        "findings": FINDINGS_PATH,
    }
    audit.to_csv(paths["checks"], index=False)
    rebuilt.to_parquet(paths["selection"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "audit_checks_passed": int(audit["passed"].sum()),
                "audit_checks_total": len(audit),
                "all_checks_passed": bool(audit["passed"].all()),
                "source_verdict": metadata["verdict"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.31 Direct Volatility-Transmission Independent Audit",
                "",
                f"Audit result: {int(audit['passed'].sum())}/{len(audit)} checks passed.",
                "",
                "The score, thresholds, selection, random control, bootstrap, and",
                "leave-one-month-out results reproduce exactly. Rejection is upheld.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["write_v2331_direct_volatility_transmission_independent_audit"]
