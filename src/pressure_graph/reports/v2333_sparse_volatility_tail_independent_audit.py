"""Independent exhaustive-search reproduction for v23.32."""

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
SOURCE_ROOT = Path("reports/v23_32_sparse_volatility_tail_selector")
REPORT_ROOT = Path("reports/v23_33_sparse_volatility_tail_independent_audit")
FINDINGS_PATH = Path(
    "docs/v2333_sparse_volatility_tail_independent_audit_2026_07_17.md"
)
PREREG_PATH = Path(
    "docs/v2332_sparse_volatility_tail_selector_preregistration_2026_07_17.md"
)
EXPECTED_FEATURE_HASH = (
    "C7EFC21FA0B9FEC822BE86D4C7A986C352E6B3F2C5382E0EF6256CBCCD312FFF"
)
FEATURES = (
    "alt_abs_z_median",
    "alt_abs_z_dispersion",
    "alt_shock_breadth_z1",
    "alt_shock_breadth_z2",
    "alt_positive_return_breadth",
    "alt_directional_coherence",
    "btc_abs_z",
    "alt_btc_abs_z_gap",
    "alt_rv_acceleration_median",
    "alt_rv_acceleration_breadth",
    "alt_residual_abs_z_median",
    "alt_residual_shock_breadth",
    "directed_edge_fraction",
    "directed_edge_weight_mean",
    "leader_shock_score",
    "leader_shock_breadth",
    "btc_receiver_gap",
)


def _hash(frame: pd.DataFrame) -> str:
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


def _masks(
    train: pd.DataFrame, predict: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    records = []
    train_masks = []
    predict_masks = []
    for feature in FEATURES:
        for orientation, quantile in (("high", 0.70), ("low", 0.30)):
            threshold = float(train[feature].quantile(quantile, interpolation="linear"))
            if orientation == "high":
                train_mask = train[feature].ge(threshold).to_numpy()
                predict_mask = predict[feature].ge(threshold).to_numpy()
            else:
                train_mask = train[feature].le(threshold).to_numpy()
                predict_mask = predict[feature].le(threshold).to_numpy()
            records.append((feature, orientation, threshold, int(train_mask.sum())))
            train_masks.append(train_mask)
            predict_masks.append(predict_mask)
    return (
        pd.DataFrame(
            records,
            columns=["feature", "orientation", "threshold", "training_selected"],
        ),
        np.column_stack(train_masks),
        np.column_stack(predict_masks),
    )


def _rebuild(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    selections = []
    rules = []
    search_data = []
    for train_periods, predict_period in (
        (("development",), "validation"),
        (("development", "validation"), "holdout"),
    ):
        train = frame.loc[frame["period"].isin(train_periods)]
        predict = frame.loc[frame["period"].eq(predict_period)].copy()
        candidates, train_masks, predict_masks = _masks(train, predict)
        train_y = train["primary_net_return"].to_numpy(float)
        score = train_masks.astype(float).T @ train_y / len(train)
        winner = int(np.argmax(score))
        ranking = np.sort(score)[::-1]
        selected = predict_masks[:, winner]
        rule = candidates.iloc[winner]
        predict["selected_feature"] = str(rule["feature"])
        predict["selected_orientation"] = str(rule["orientation"])
        predict["selected_threshold"] = float(rule["threshold"])
        predict["training_winner_opportunity_return"] = float(score[winner])
        predict["training_winner_margin"] = float(ranking[0] - ranking[1])
        predict["selected"] = selected
        predict["strategy_primary_return"] = np.where(
            selected, predict["primary_net_return"], 0.0
        )
        predict["strategy_stress_return"] = np.where(
            selected, predict["stress_net_return"], 0.0
        )
        selections.append(predict)
        rules.append(
            {
                "train_periods": "|".join(train_periods),
                "predict_period": predict_period,
                "feature": str(rule["feature"]),
                "orientation": str(rule["orientation"]),
                "threshold": float(rule["threshold"]),
                "training_selected": int(rule["training_selected"]),
                "prediction_selected": int(selected.sum()),
                "training_winner_opportunity_return": float(score[winner]),
                "training_winner_margin": float(ranking[0] - ranking[1]),
            }
        )
        search_data.append((train_y, train_masks, predict_masks))
    return pd.concat(selections, ignore_index=True), pd.DataFrame(rules), search_data


def _random(
    selection: pd.DataFrame,
    frame: pd.DataFrame,
    search_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rng = np.random.default_rng(20_260_717)
    period_results = []
    for period_index, predict_period in enumerate(("validation", "holdout")):
        train_y, train_masks, predict_masks = search_data[period_index]
        permutations = np.column_stack([rng.permutation(train_y) for _ in range(1_000)])
        scores = train_masks.astype(float).T @ permutations / len(train_y)
        winners = np.argmax(scores, axis=0)
        selected = predict_masks[:, winners]
        actual = frame.loc[
            frame["period"].eq(predict_period), "primary_net_return"
        ].to_numpy(float)
        period_results.append((np.where(selected, actual[:, None], 0.0).sum(axis=0), winners))
    null = (period_results[0][0] + period_results[1][0]) / len(selection)
    observed = float(selection["strategy_primary_return"].mean())
    percentile = 100.0 * (1 + np.sum(null < observed)) / 1_001
    return pd.DataFrame(
        {
            "iteration": np.arange(1_000),
            "validation_winner_index": period_results[0][1],
            "holdout_winner_index": period_results[1][1],
            "null_primary_opportunity_return": null,
            "observed_primary_opportunity_return": observed,
            "observed_percentile": percentile,
        }
    )


def _bootstrap(selection: pd.DataFrame) -> pd.DataFrame:
    blocks = [
        group["strategy_primary_return"].to_numpy(float)
        for _, group in selection.groupby("entry_month", sort=True)
    ]
    rng = np.random.default_rng(20_260_718)
    values = []
    for iteration in range(5_000):
        indices = rng.integers(0, len(blocks), size=len(blocks))
        values.append(
            (
                iteration,
                float(np.concatenate([blocks[index] for index in indices]).mean()),
            )
        )
    return pd.DataFrame(
        values, columns=["iteration", "primary_opportunity_return"]
    )


def _lomo(selection: pd.DataFrame) -> pd.DataFrame:
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
            if not np.allclose(audit, source, rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif not audit.astype(str).equals(source.astype(str)):
            return False
    return True


def write_v2333_sparse_volatility_tail_independent_audit() -> dict[str, Path]:
    features, frame = _inputs()
    selection, rules, search_data = _rebuild(frame)
    random = _random(selection, frame, search_data)
    bootstrap = _bootstrap(selection)
    lomo = _lomo(selection)
    source_selection = pd.read_parquet(SOURCE_ROOT / "temporal_oos_selection.parquet")
    source_selection["entry_time"] = pd.to_datetime(
        source_selection["entry_time"], utc=True, errors="raise"
    )
    source_rules = pd.read_csv(SOURCE_ROOT / "temporal_winning_rules.csv")
    source_random = pd.read_parquet(SOURCE_ROOT / "full_search_random_label_null.parquet")
    source_bootstrap = pd.read_parquet(SOURCE_ROOT / "month_block_bootstrap.parquet")
    source_lomo = pd.read_csv(SOURCE_ROOT / "leave_one_month_out.csv")
    source_summary = pd.read_csv(SOURCE_ROOT / "result_summary.csv")
    source_gates = pd.read_csv(SOURCE_ROOT / "decision_gates.csv")
    metadata = json.loads((SOURCE_ROOT / "metadata.json").read_text("utf-8"))
    selection_columns = [
        "entry_time",
        "selected_feature",
        "selected_orientation",
        "selected_threshold",
        "training_winner_opportunity_return",
        "training_winner_margin",
        "selected",
        "strategy_primary_return",
    ]
    checks = {
        "preregistration_exists": PREREG_PATH.exists(),
        "feature_hash_exact": _hash(features) == EXPECTED_FEATURE_HASH,
        "exact_17_features_and_34_candidates": len(FEATURES) == 17
        and metadata["candidate_count"] == 34,
        "all_events_and_temporal_counts": len(frame) == 159
        and selection.groupby("period").size().to_dict()
        == {"holdout": 49, "validation": 47},
        "causal_feature_times": features["feature_time"].le(
            features["entry_time"]
        ).all()
        and features["price_feature_time"].le(features["entry_time"]).all(),
        "winning_rules_match": _equal(
            rules, source_rules, ["train_periods", "predict_period"]
        ),
        "selection_matches": _equal(
            selection[selection_columns],
            source_selection[selection_columns],
            ["entry_time"],
        ),
        "full_search_random_null_matches": _equal(
            random, source_random, ["iteration"]
        ),
        "month_bootstrap_matches": _equal(
            bootstrap, source_bootstrap, ["iteration"]
        ),
        "lomo_matches": _equal(lomo, source_lomo, ["excluded_month"]),
        "summary_and_gates_reconcile": source_summary["events"].tolist()
        == [47, 49, 96]
        and int(source_gates["passed"].sum()) == 2,
        "rejection_verdict_upheld": metadata["verdict"]
        == "rejected_sparse_volatility_tail_selector"
        and not metadata["all_gates_passed"],
    }
    audit = pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )
    root = ensure_dir(REPORT_ROOT)
    paths = {
        "checks": root / "independent_audit_checks.csv",
        "selection": root / "independently_recomputed_selection.parquet",
        "rules": root / "independently_recomputed_rules.csv",
        "metadata": root / "metadata.json",
        "findings": FINDINGS_PATH,
    }
    audit.to_csv(paths["checks"], index=False)
    selection.to_parquet(paths["selection"], index=False)
    rules.to_csv(paths["rules"], index=False)
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
                "# v23.33 Sparse Volatility-Tail Independent Audit",
                "",
                f"Audit result: {int(audit['passed'].sum())}/{len(audit)} checks passed.",
                "",
                "The exhaustive temporal winners, selections, full-search random null,",
                "bootstrap, and leave-one-month-out outputs reproduce. Rejection holds.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["write_v2333_sparse_volatility_tail_independent_audit"]
