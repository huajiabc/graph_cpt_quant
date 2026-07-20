"""Temporally refit one-feature tail selector with full-search random control."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2329_event_volatility_transmission_feature_audit import (
    VOLATILITY_FEATURES,
    feature_hash_v2329,
)


FEATURE_PATH = Path(
    "reports/v23_29_event_volatility_transmission_feature_audit/"
    "event_volatility_transmission_features.parquet"
)
OUTCOME_PATH = Path(
    "reports/v23_4_book_vacuum_oco_breakout/barrier_variant_outcomes.parquet"
)
REPORT_ROOT = Path("reports/v23_32_sparse_volatility_tail_selector")
FINDINGS_PATH = Path("docs/v2332_sparse_volatility_tail_selector_2026_07_17.md")
PREREG_PATH = Path(
    "docs/v2332_sparse_volatility_tail_selector_preregistration_2026_07_17.md"
)
CANDIDATE = "SVT1_SPARSE_VOLATILITY_TAIL_OCO_SELECTOR"
EXPECTED_FEATURE_HASH = (
    "C7EFC21FA0B9FEC822BE86D4C7A986C352E6B3F2C5382E0EF6256CBCCD312FFF"
)


@dataclass(frozen=True)
class V2332Config:
    feature_path: Path = FEATURE_PATH
    outcome_path: Path = OUTCOME_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    sigma_multiple: float = 0.75
    upper_quantile: float = 0.70
    lower_quantile: float = 0.30
    minimum_period_trades: int = 8
    minimum_active_months: int = 5
    minimum_positive_month_fraction: float = 0.60
    permutation_iterations: int = 1_000
    bootstrap_iterations: int = 5_000
    seed: int = 20_260_717


def load_v2332_inputs(
    cfg: V2332Config = V2332Config(),
) -> tuple[pd.DataFrame, str]:
    features = pd.read_parquet(cfg.feature_path)
    for column in ("feature_time", "entry_time", "price_feature_time"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="raise")
    feature_hash = feature_hash_v2329(features)
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
    frame = features.merge(outcomes, on="entry_time", validate="one_to_one")
    if len(frame) != 159:
        raise ValueError("feature/outcome join must retain all 159 events")
    return frame.sort_values("entry_time").reset_index(drop=True), feature_hash


def _candidate_matrices(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    cfg: V2332Config,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    records = []
    train_masks = []
    predict_masks = []
    for feature in VOLATILITY_FEATURES:
        train_values = train[feature].to_numpy(float)
        predict_values = predict[feature].to_numpy(float)
        for orientation, quantile in (
            ("high", cfg.upper_quantile),
            ("low", cfg.lower_quantile),
        ):
            threshold = float(np.quantile(train_values, quantile))
            if orientation == "high":
                train_mask = train_values >= threshold
                predict_mask = predict_values >= threshold
            else:
                train_mask = train_values <= threshold
                predict_mask = predict_values <= threshold
            records.append(
                {
                    "feature": feature,
                    "orientation": orientation,
                    "threshold": threshold,
                    "training_selected": int(train_mask.sum()),
                }
            )
            train_masks.append(train_mask)
            predict_masks.append(predict_mask)
    return (
        pd.DataFrame(records),
        np.column_stack(train_masks),
        np.column_stack(predict_masks),
    )


def _fit_period(
    frame: pd.DataFrame,
    train_periods: tuple[str, ...],
    predict_period: str,
    cfg: V2332Config,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = frame.loc[frame["period"].isin(train_periods)]
    predict = frame.loc[frame["period"].eq(predict_period)].copy()
    candidates, train_masks, predict_masks = _candidate_matrices(train, predict, cfg)
    train_y = train["primary_net_return"].to_numpy(float)
    scores = train_masks.astype(float).T @ train_y / len(train_y)
    winner = int(np.argmax(scores))
    ranking = np.sort(scores)[::-1]
    selected = predict_masks[:, winner]
    rule = candidates.iloc[winner]
    predict["selected_feature"] = str(rule["feature"])
    predict["selected_orientation"] = str(rule["orientation"])
    predict["selected_threshold"] = float(rule["threshold"])
    predict["training_winner_opportunity_return"] = float(scores[winner])
    predict["training_winner_margin"] = float(ranking[0] - ranking[1])
    predict["selected"] = selected
    predict["strategy_primary_return"] = np.where(
        selected, predict["primary_net_return"], 0.0
    )
    predict["strategy_stress_return"] = np.where(
        selected, predict["stress_net_return"], 0.0
    )
    return predict, {
        "train_periods": "|".join(train_periods),
        "predict_period": predict_period,
        "feature": str(rule["feature"]),
        "orientation": str(rule["orientation"]),
        "threshold": float(rule["threshold"]),
        "training_selected": int(rule["training_selected"]),
        "prediction_selected": int(selected.sum()),
        "training_winner_opportunity_return": float(scores[winner]),
        "training_winner_margin": float(ranking[0] - ranking[1]),
    }


def build_v2332_selection(
    frame: pd.DataFrame,
    cfg: V2332Config = V2332Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation, validation_rule = _fit_period(
        frame, ("development",), "validation", cfg
    )
    holdout, holdout_rule = _fit_period(
        frame, ("development", "validation"), "holdout", cfg
    )
    return (
        pd.concat([validation, holdout], ignore_index=True),
        pd.DataFrame([validation_rule, holdout_rule]),
    )


def summarize_v2332(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("validation", "holdout", "oos"):
        local = (
            selection
            if scope == "oos"
            else selection.loc[selection["period"].eq(scope)]
        )
        selected = local.loc[local["selected"]]
        rows.append(
            {
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
            }
        )
    return pd.DataFrame(rows)


def random_search_control_v2332(
    frame: pd.DataFrame,
    selection: pd.DataFrame,
    cfg: V2332Config = V2332Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    period_data = []
    for train_periods, predict_period in (
        (("development",), "validation"),
        (("development", "validation"), "holdout"),
    ):
        train = frame.loc[frame["period"].isin(train_periods)]
        predict = frame.loc[frame["period"].eq(predict_period)]
        candidates, train_masks, predict_masks = _candidate_matrices(
            train, predict, cfg
        )
        del candidates
        train_y = train["primary_net_return"].to_numpy(float)
        permutations = np.column_stack(
            [rng.permutation(train_y) for _ in range(cfg.permutation_iterations)]
        )
        scores = train_masks.astype(float).T @ permutations / len(train)
        winners = np.argmax(scores, axis=0)
        selected = predict_masks[:, winners]
        actual_y = predict["primary_net_return"].to_numpy(float)
        strategy = np.where(selected, actual_y[:, None], 0.0)
        period_data.append((strategy.sum(axis=0), winners))
    null_score = (period_data[0][0] + period_data[1][0]) / len(selection)
    observed = float(selection["strategy_primary_return"].mean())
    percentile = 100.0 * (1 + np.sum(null_score < observed)) / (
        len(null_score) + 1
    )
    return pd.DataFrame(
        {
            "iteration": np.arange(cfg.permutation_iterations),
            "validation_winner_index": period_data[0][1],
            "holdout_winner_index": period_data[1][1],
            "null_primary_opportunity_return": null_score,
            "observed_primary_opportunity_return": observed,
            "observed_percentile": percentile,
        }
    )


def month_diagnostics_v2332(selection: pd.DataFrame) -> pd.DataFrame:
    return (
        selection.groupby("entry_month", sort=True)
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


def bootstrap_v2332(
    selection: pd.DataFrame,
    cfg: V2332Config = V2332Config(),
) -> pd.DataFrame:
    blocks = [
        group["strategy_primary_return"].to_numpy(float)
        for _, group in selection.groupby("entry_month", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
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


def lomo_v2332(selection: pd.DataFrame) -> pd.DataFrame:
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


def gates_v2332(
    summary: pd.DataFrame,
    rules: pd.DataFrame,
    months: pd.DataFrame,
    random_control: pd.DataFrame,
    bootstrap: pd.DataFrame,
    lomo: pd.DataFrame,
    cfg: V2332Config = V2332Config(),
) -> pd.DataFrame:
    def value(scope: str, column: str) -> float:
        return float(summary.loc[summary["scope"].eq(scope), column].iloc[0])

    active = months.loc[months["selected_trades"].gt(0)]
    checks = {
        "minimum_8_trades_each_period": all(
            value(scope, "selected_trades") >= cfg.minimum_period_trades
            for scope in ("validation", "holdout")
        ),
        "positive_primary_selected_each_period": all(
            value(scope, "primary_selected_expectancy_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "positive_stress_selected_each_period": all(
            value(scope, "stress_selected_expectancy_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "positive_primary_opportunity_each_period": all(
            value(scope, "primary_opportunity_return_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "positive_stress_opportunity_each_period": all(
            value(scope, "stress_opportunity_return_bp") > 0
            for scope in ("validation", "holdout")
        ),
        "beats_unfiltered_all_scopes": all(
            value(scope, "primary_opportunity_return_bp")
            > value(scope, "unfiltered_primary_return_bp")
            for scope in ("validation", "holdout", "oos")
        ),
        "same_rule_wins_both_temporal_fits": rules["feature"].nunique() == 1
        and rules["orientation"].nunique() == 1,
        "full_search_random_percentile_at_least_95": float(
            random_control["observed_percentile"].iloc[0]
        )
        >= 95.0,
        "month_bootstrap_q05_positive": float(
            bootstrap["primary_opportunity_return"].quantile(0.05)
        )
        > 0,
        "month_and_lomo_stability": len(active) >= cfg.minimum_active_months
        and float(active["primary_return"].gt(0).mean())
        >= cfg.minimum_positive_month_fraction
        and lomo["primary_opportunity_return"].gt(0).all(),
    }
    return pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in checks.items()]
    )


def write_v2332_sparse_volatility_tail_selector(
    cfg: V2332Config = V2332Config(),
) -> dict[str, Path]:
    frame, feature_hash = load_v2332_inputs(cfg)
    selection, rules = build_v2332_selection(frame, cfg)
    summary = summarize_v2332(selection)
    random_control = random_search_control_v2332(frame, selection, cfg)
    months = month_diagnostics_v2332(selection)
    bootstrap = bootstrap_v2332(selection, cfg)
    lomo = lomo_v2332(selection)
    gates = gates_v2332(summary, rules, months, random_control, bootstrap, lomo, cfg)
    passed = bool(gates["passed"].all())
    verdict = (
        "research_candidate_requires_isolated_forward_shadow"
        if passed
        else "rejected_sparse_volatility_tail_selector"
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "selection": root / "temporal_oos_selection.parquet",
        "rules": root / "temporal_winning_rules.csv",
        "summary": root / "result_summary.csv",
        "random_control": root / "full_search_random_label_null.parquet",
        "months": root / "oos_month_diagnostics.csv",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "lomo": root / "leave_one_month_out.csv",
        "gates": root / "decision_gates.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    selection.to_parquet(paths["selection"], index=False)
    rules.to_csv(paths["rules"], index=False)
    summary.to_csv(paths["summary"], index=False)
    random_control.to_parquet(paths["random_control"], index=False)
    months.to_csv(paths["months"], index=False)
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
                "eligible_features": list(VOLATILITY_FEATURES),
                "candidate_count": 2 * len(VOLATILITY_FEATURES),
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
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.32 Sparse Volatility-Tail Selector",
                "",
                f"Verdict: `{verdict}`.",
                "",
                rules.to_markdown(index=False, floatfmt=".6f"),
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                f"Passed gates: {int(gates['passed'].sum())}/{len(gates)}.",
                f"Full-search random-label percentile: {random_control['observed_percentile'].iloc[0]:.2f}.",
                "Month-bootstrap q05 (bp/opportunity): "
                f"{bootstrap['primary_opportunity_return'].quantile(0.05) * 10_000:.4f}.",
                "",
                "The random control repeats the complete 34-candidate search.",
                "No PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2332Config",
    "build_v2332_selection",
    "gates_v2332",
    "load_v2332_inputs",
    "write_v2332_sparse_volatility_tail_selector",
]
