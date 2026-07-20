"""Outcome-free continuous volatility-transmission score on fixed OCO payoffs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2329_event_volatility_transmission_feature_audit import (
    feature_hash_v2329,
)


FEATURE_PATH = Path(
    "reports/v23_29_event_volatility_transmission_feature_audit/"
    "event_volatility_transmission_features.parquet"
)
OUTCOME_PATH = Path(
    "reports/v23_4_book_vacuum_oco_breakout/barrier_variant_outcomes.parquet"
)
REPORT_ROOT = Path("reports/v23_30_direct_volatility_transmission_selector")
FINDINGS_PATH = Path(
    "docs/v2330_direct_volatility_transmission_selector_2026_07_17.md"
)
PREREG_PATH = Path(
    "docs/v2330_direct_volatility_transmission_selector_preregistration_2026_07_17.md"
)
CANDIDATE = "EVT1_DIRECT_VOLATILITY_TRANSMISSION_OCO_SELECTOR"
EXPECTED_FEATURE_HASH = (
    "C7EFC21FA0B9FEC822BE86D4C7A986C352E6B3F2C5382E0EF6256CBCCD312FFF"
)
SCORE_FEATURES = (
    "btc_receiver_gap",
    "alt_rv_acceleration_median",
    "alt_residual_shock_breadth",
    "directed_edge_weight_mean",
)


@dataclass(frozen=True)
class V2330Config:
    feature_path: Path = FEATURE_PATH
    outcome_path: Path = OUTCOME_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    sigma_multiple: float = 0.75
    score_quantile: float = 0.70
    minimum_period_trades: int = 8
    minimum_active_months: int = 5
    minimum_positive_month_fraction: float = 0.60
    permutation_iterations: int = 1_000
    bootstrap_iterations: int = 5_000
    seed: int = 20_260_717


def load_v2330_inputs(
    cfg: V2330Config = V2330Config(),
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


def _score_period(
    frame: pd.DataFrame,
    train_periods: tuple[str, ...],
    predict_period: str,
    cfg: V2330Config,
) -> pd.DataFrame:
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
    threshold = float(np.quantile(train_score, cfg.score_quantile))
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
    return predict


def build_v2330_selection(
    frame: pd.DataFrame,
    cfg: V2330Config = V2330Config(),
) -> pd.DataFrame:
    validation = _score_period(frame, ("development",), "validation", cfg)
    holdout = _score_period(
        frame, ("development", "validation"), "holdout", cfg
    )
    return pd.concat([validation, holdout], ignore_index=True)


def summarize_v2330(selection: pd.DataFrame) -> pd.DataFrame:
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
                "score_primary_spearman": float(
                    local["transmission_score"].corr(
                        local["primary_net_return"], method="spearman"
                    )
                ),
                "mean_selected_score": float(selected["transmission_score"].mean()),
                "mean_unselected_score": float(
                    local.loc[~local["selected"], "transmission_score"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def month_diagnostics_v2330(selection: pd.DataFrame) -> pd.DataFrame:
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


def random_control_v2330(
    selection: pd.DataFrame,
    cfg: V2330Config = V2330Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    validation = selection.loc[selection["period"].eq("validation")]
    holdout = selection.loc[selection["period"].eq("holdout")]
    validation_n = int(validation["selected"].sum())
    holdout_n = int(holdout["selected"].sum())
    validation_y = validation["primary_net_return"].to_numpy(float)
    holdout_y = holdout["primary_net_return"].to_numpy(float)
    rows = []
    for iteration in range(cfg.permutation_iterations):
        validation_index = rng.choice(
            len(validation_y), size=validation_n, replace=False
        )
        holdout_index = rng.choice(len(holdout_y), size=holdout_n, replace=False)
        score = (
            validation_y[validation_index].sum() + holdout_y[holdout_index].sum()
        ) / (len(validation_y) + len(holdout_y))
        rows.append((iteration, float(score)))
    output = pd.DataFrame(
        rows, columns=["iteration", "null_primary_opportunity_return"]
    )
    observed = float(selection["strategy_primary_return"].mean())
    output["observed_primary_opportunity_return"] = observed
    output["observed_percentile"] = (
        100.0
        * (1 + output["null_primary_opportunity_return"].lt(observed).sum())
        / (len(output) + 1)
    )
    return output


def bootstrap_v2330(
    selection: pd.DataFrame,
    cfg: V2330Config = V2330Config(),
) -> pd.DataFrame:
    blocks = [
        group["strategy_primary_return"].to_numpy(float)
        for _, group in selection.groupby("entry_month", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
        indices = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[index] for index in indices])
        rows.append((iteration, float(sample.mean())))
    return pd.DataFrame(
        rows, columns=["iteration", "primary_opportunity_return"]
    )


def lomo_v2330(selection: pd.DataFrame) -> pd.DataFrame:
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


def gates_v2330(
    summary: pd.DataFrame,
    months: pd.DataFrame,
    random_control: pd.DataFrame,
    bootstrap: pd.DataFrame,
    lomo: pd.DataFrame,
    cfg: V2330Config = V2330Config(),
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
        "positive_score_spearman_each_period": all(
            value(scope, "score_primary_spearman") > 0
            for scope in ("validation", "holdout")
        ),
        "random_same_count_percentile_at_least_95": float(
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


def write_v2330_direct_volatility_transmission_selector(
    cfg: V2330Config = V2330Config(),
) -> dict[str, Path]:
    frame, feature_hash = load_v2330_inputs(cfg)
    selection = build_v2330_selection(frame, cfg)
    summary = summarize_v2330(selection)
    months = month_diagnostics_v2330(selection)
    random_control = random_control_v2330(selection, cfg)
    bootstrap = bootstrap_v2330(selection, cfg)
    lomo = lomo_v2330(selection)
    gates = gates_v2330(summary, months, random_control, bootstrap, lomo, cfg)
    passed = bool(gates["passed"].all())
    verdict = (
        "research_candidate_requires_isolated_forward_shadow"
        if passed
        else "rejected_direct_volatility_transmission_selector"
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "selection": root / "temporal_oos_selection.parquet",
        "summary": root / "result_summary.csv",
        "months": root / "oos_month_diagnostics.csv",
        "random_control": root / "random_same_count_control.parquet",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "lomo": root / "leave_one_month_out.csv",
        "gates": root / "decision_gates.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    selection.to_parquet(paths["selection"], index=False)
    summary.to_csv(paths["summary"], index=False)
    months.to_csv(paths["months"], index=False)
    random_control.to_parquet(paths["random_control"], index=False)
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
                "score_features": list(SCORE_FEATURES),
                "outcome_trained": False,
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
                "# v23.30 Direct Volatility-Transmission Selector",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                f"Passed gates: {int(gates['passed'].sum())}/{len(gates)}.",
                f"Random same-count percentile: {random_control['observed_percentile'].iloc[0]:.2f}.",
                "Month-bootstrap q05 (bp/opportunity): "
                f"{bootstrap['primary_opportunity_return'].quantile(0.05) * 10_000:.4f}.",
                "",
                "The selection score and cutoff are outcome-free.",
                "No PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "SCORE_FEATURES",
    "V2330Config",
    "build_v2330_selection",
    "gates_v2330",
    "load_v2330_inputs",
    "summarize_v2330",
    "write_v2330_direct_volatility_transmission_selector",
]
