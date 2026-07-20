"""Adaptive walk-forward graph-ML audit for bearish convergence spreads."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v141_directed_taker_flow_graph import (
    load_v141_price_matrices,
)
from pressure_graph.reports.v142_community_volatility_transmission import (
    V142Config,
    load_v142_membership,
)
from pressure_graph.reports.v143_quiet_receiver_convergence import (
    MEMBERSHIP_PATH,
    V143Config,
    build_v143_graph_and_contexts,
    reverse_v143_edges,
)


REPORT_ROOT = Path("reports/v14_4_bearish_graph_ml_convergence")
FINDINGS_PATH = Path(
    "docs/v144_bearish_graph_ml_convergence_findings_2026_07_15.md"
)
CANDIDATE = "GML1_BEARISH_VOL_CONVERGENCE_12H"
FEATURE_COLUMNS = (
    "source_return_z",
    "source_volatility_z",
    "source_breadth",
    "receiver_return_z",
    "receiver_volatility_z",
    "receiver_breadth",
    "edge_weight",
    "volatility_spearman",
    "magnitude_advantage",
    "return_z_gap",
)


@dataclass(frozen=True)
class V144Config:
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    bearish_return_z_max: float = -1.5
    source_volatility_z_min: float = 0.5
    source_breadth_min: float = 0.60
    receiver_abs_return_z_max: float = 1.25
    receiver_volatility_z_max: float = 0.5
    min_node_members: int = 5
    horizon_hours: int = 12
    cooldown_hours: int = 12
    prediction_hurdle: float = 0.004
    minimum_training_rows: int = 200
    minimum_training_days: int = 50
    delayed_state_hours: int = 24
    label_null_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def _month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values.dt.strftime("%Y-%m-01"), utc=True, errors="coerce"
    )


def _state_frames(
    context: dict,
    shift_hours: int,
) -> dict[str, pd.DataFrame]:
    names = (
        "node_return_z",
        "node_volatility_z",
        "node_breadth",
        "node_member_count",
    )
    return {
        name: context[name].shift(shift_hours) if shift_hours else context[name]
        for name in names
    }


def build_v144_pair_panel(
    contexts: dict[pd.Timestamp, dict],
    edges: pd.DataFrame,
    cfg: V144Config,
    state_shift_hours: int = 0,
) -> pd.DataFrame:
    rows = []
    for month, context in sorted(contexts.items()):
        month_edges = edges[edges["month_start"].eq(month)]
        if month_edges.empty:
            continue
        state = _state_frames(context, state_shift_hours)
        raw_future = context["raw_future"][cfg.horizon_hours]
        residual_future = context["residual_future"][cfg.horizon_hours]
        for timestamp in state["node_return_z"].index:
            return_z = state["node_return_z"].loc[timestamp]
            volatility_z = state["node_volatility_z"].loc[timestamp]
            breadth = state["node_breadth"].loc[timestamp]
            member_count = state["node_member_count"].loc[timestamp]
            active_sources = return_z.index[
                return_z.le(cfg.bearish_return_z_max)
                & volatility_z.ge(cfg.source_volatility_z_min)
                & breadth.ge(cfg.source_breadth_min)
                & member_count.ge(cfg.min_node_members)
            ]
            if active_sources.empty:
                continue
            quiet_receivers = set(
                return_z.index[
                    return_z.abs().le(cfg.receiver_abs_return_z_max)
                    & volatility_z.le(cfg.receiver_volatility_z_max)
                    & member_count.ge(cfg.min_node_members)
                ]
            )
            for source in active_sources:
                local_edges = month_edges[
                    month_edges["leader_community"].eq(source)
                    & month_edges["follower_community"].isin(quiet_receivers)
                ]
                if local_edges.empty:
                    continue
                source_members = context["community_members"][source]
                source_raw = pd.to_numeric(
                    raw_future.loc[timestamp, source_members], errors="coerce"
                ).dropna()
                source_residual = pd.to_numeric(
                    residual_future.loc[timestamp, source_members], errors="coerce"
                ).dropna()
                source_valid = source_raw.index.intersection(source_residual.index)
                if len(source_valid) < cfg.min_node_members:
                    continue
                for edge in local_edges.itertuples(index=False):
                    receiver = str(edge.follower_community)
                    receiver_members = context["community_members"][receiver]
                    receiver_raw = pd.to_numeric(
                        raw_future.loc[timestamp, receiver_members], errors="coerce"
                    ).dropna()
                    receiver_residual = pd.to_numeric(
                        residual_future.loc[timestamp, receiver_members],
                        errors="coerce",
                    ).dropna()
                    receiver_valid = receiver_raw.index.intersection(
                        receiver_residual.index
                    )
                    if len(receiver_valid) < cfg.min_node_members:
                        continue
                    raw_gross = 0.5 * (
                        float(source_raw[source_valid].mean())
                        - float(receiver_raw[receiver_valid].mean())
                    )
                    residual_gross = 0.5 * (
                        float(source_residual[source_valid].mean())
                        - float(receiver_residual[receiver_valid].mean())
                    )
                    rows.append(
                        {
                            "feature_time": timestamp,
                            "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                            "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                            "month_start": month,
                            "period": context["period"],
                            "source_community": source,
                            "receiver_community": receiver,
                            "source_return_z": float(return_z[source]),
                            "source_volatility_z": float(volatility_z[source]),
                            "source_breadth": float(breadth[source]),
                            "receiver_return_z": float(return_z[receiver]),
                            "receiver_volatility_z": float(volatility_z[receiver]),
                            "receiver_breadth": float(breadth[receiver]),
                            "edge_weight": float(edge.edge_weight),
                            "volatility_spearman": float(edge.volatility_spearman),
                            "magnitude_advantage": float(edge.magnitude_advantage),
                            "return_z_gap": float(
                                return_z[source] - return_z[receiver]
                            ),
                            "raw_gross_12h": raw_gross,
                            "residual_gross_12h": residual_gross,
                            "target_ready_time": pd.Timestamp(timestamp)
                            + pd.Timedelta(hours=cfg.horizon_hours),
                        }
                    )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    return (
        panel.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[*FEATURE_COLUMNS, "raw_gross_12h", "residual_gross_12h"])
        .sort_values(["feature_time", "source_community", "receiver_community"])
        .reset_index(drop=True)
    )


def permute_training_labels(
    training: pd.DataFrame,
    iteration: int,
    cfg: V144Config,
) -> np.ndarray:
    labels = training["residual_gross_12h"].to_numpy(dtype=float).copy()
    rng = np.random.default_rng(cfg.seed + iteration * 1009)
    months = training["entry_month"].astype(str).to_numpy()
    for month in sorted(set(months)):
        indices = np.flatnonzero(months == month)
        labels[indices] = rng.permutation(labels[indices])
    return labels


def walk_forward_v144_predictions(
    panel: pd.DataFrame,
    cfg: V144Config,
    label_permutation_iteration: int | None = None,
) -> pd.DataFrame:
    predictions = []
    if panel.empty:
        return panel.assign(predicted_residual_gross=np.nan)
    for raw_month in sorted(panel["month_start"].unique()):
        month = pd.Timestamp(raw_month)
        deployment = panel[panel["month_start"].eq(month)].copy()
        training = panel[panel["target_ready_time"].lt(month)].copy()
        if (
            len(training) < cfg.minimum_training_rows
            or training["entry_day"].nunique() < cfg.minimum_training_days
            or deployment.empty
        ):
            continue
        labels = (
            training["residual_gross_12h"].to_numpy(dtype=float)
            if label_permutation_iteration is None
            else permute_training_labels(
                training, label_permutation_iteration, cfg
            )
        )
        counts = training.groupby("feature_time")["feature_time"].transform("size")
        weights = 1.0 / counts.to_numpy(dtype=float)
        model = HistGradientBoostingRegressor(
            max_depth=2,
            learning_rate=0.05,
            max_iter=100,
            min_samples_leaf=20,
            l2_regularization=5.0,
            random_state=cfg.seed,
        )
        model.fit(
            training.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float),
            labels,
            sample_weight=weights,
        )
        deployment["predicted_residual_gross"] = model.predict(
            deployment.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        )
        deployment["training_rows"] = len(training)
        deployment["training_days"] = training["entry_day"].nunique()
        deployment["training_max_ready_time"] = training[
            "target_ready_time"
        ].max()
        predictions.append(deployment)
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def build_v144_model_portfolio(
    predictions: pd.DataFrame,
    cfg: V144Config,
    score_column: str = "predicted_residual_gross",
    score_threshold: float | None = None,
) -> pd.DataFrame:
    columns = (
        "candidate",
        "feature_time",
        "entry_day",
        "entry_month",
        "period",
        "source_community",
        "receiver_community",
        "score",
        "training_rows",
        "raw_gross_12h",
        "raw_net_12h_20bp",
        "raw_net_12h_30bp",
        "residual_gross_12h",
        "residual_net_12h_40bp",
    )
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    threshold = cfg.prediction_hurdle if score_threshold is None else score_threshold
    eligible = predictions[predictions[score_column].ge(threshold)].copy()
    rows = []
    last: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for timestamp, group in eligible.groupby("feature_time", sort=True):
        timestamp = pd.Timestamp(timestamp)
        if last is not None and timestamp - last < cooldown:
            continue
        selected = group.sort_values(score_column, ascending=False).iloc[0]
        raw_gross = float(selected["raw_gross_12h"])
        residual_gross = float(selected["residual_gross_12h"])
        rows.append(
            {
                "candidate": CANDIDATE,
                "feature_time": timestamp,
                "entry_day": selected["entry_day"],
                "entry_month": selected["entry_month"],
                "period": selected["period"],
                "source_community": selected["source_community"],
                "receiver_community": selected["receiver_community"],
                "score": float(selected[score_column]),
                "training_rows": int(selected.get("training_rows", 0)),
                "raw_gross_12h": raw_gross,
                "raw_net_12h_20bp": raw_gross - 0.002,
                "raw_net_12h_30bp": raw_gross - 0.003,
                "residual_gross_12h": residual_gross,
                "residual_net_12h_40bp": residual_gross - 0.004,
            }
        )
        last = timestamp
    return pd.DataFrame(rows, columns=columns)


def build_v144_static_portfolio(
    panel: pd.DataFrame,
    cfg: V144Config,
) -> pd.DataFrame:
    if panel.empty:
        return build_v144_model_portfolio(panel, cfg)
    predictions = panel.copy()
    predictions["static_score"] = predictions["edge_weight"] * predictions[
        "source_return_z"
    ].abs()
    predictions["training_rows"] = 0
    return build_v144_model_portfolio(
        predictions,
        cfg,
        score_column="static_score",
        score_threshold=-np.inf,
    )


def summarize_v144(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = (
            portfolio if scope == "all" else portfolio[portfolio["period"].eq(scope)]
        )
        rows.append(
            {
                "scope": scope,
                "candidate": CANDIDATE,
                "observations": len(sample),
                "active_days": sample["entry_day"].nunique(),
                "active_months": sample["entry_month"].nunique(),
                "mean_raw_gross_12h": sample["raw_gross_12h"].mean(),
                "mean_raw_net_12h_20bp": sample["raw_net_12h_20bp"].mean(),
                "mean_raw_net_12h_30bp": sample["raw_net_12h_30bp"].mean(),
                "mean_residual_gross_12h": sample[
                    "residual_gross_12h"
                ].mean(),
                "mean_residual_net_12h_40bp": sample[
                    "residual_net_12h_40bp"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def label_null_v144_controls(
    panel: pd.DataFrame,
    cfg: V144Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.label_null_iterations):
        predictions = walk_forward_v144_predictions(
            panel, cfg, label_permutation_iteration=iteration
        )
        portfolio = build_v144_model_portfolio(predictions, cfg)
        rows.append(
            {
                "iteration": iteration,
                "observations": len(portfolio),
                "mean_residual_net_12h_40bp": portfolio[
                    "residual_net_12h_40bp"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(
    sample: pd.DataFrame,
    cfg: V144Config,
) -> tuple[float, float]:
    daily = [
        group["residual_net_12h_40bp"].dropna().to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    daily = [values for values in daily if len(values)]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        boot.append(
            float(np.mean(np.concatenate([daily[index] for index in chosen])))
        )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _positive_share(values: pd.Series) -> float:
    positive = values.clip(lower=0.0)
    return float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf


def audit_v144(
    real: pd.DataFrame,
    summary: pd.DataFrame,
    static: pd.DataFrame,
    reversed_portfolio: pd.DataFrame,
    delayed_portfolio: pd.DataFrame,
    label_nulls: pd.DataFrame,
    cfg: V144Config,
) -> pd.DataFrame:
    lookup = {row.scope: row for row in summary.itertuples(index=False)}
    ci_low, ci_high = _bootstrap_ci(real, cfg)
    real_mean = float(lookup["all"].mean_residual_net_12h_40bp)
    static_mean = float(static["residual_net_12h_40bp"].mean())
    reversed_mean = float(reversed_portfolio["residual_net_12h_40bp"].mean())
    delayed_mean = float(delayed_portfolio["residual_net_12h_40bp"].mean())
    null_values = label_nulls["mean_residual_net_12h_40bp"].dropna()
    percentile = float(null_values.lt(real_mean).mean()) if len(null_values) else np.nan
    month_share = _positive_share(
        real.groupby("entry_month")["residual_net_12h_40bp"].sum()
    )
    worst_period = float(
        real.groupby("period")["residual_net_12h_40bp"].mean().min()
    )
    gates = {
        "full_observations_80": lookup["all"].observations >= 80,
        "validation_observations_20": lookup["validation"].observations >= 20,
        "holdout_observations_20": lookup["holdout"].observations >= 20,
        "development_residual_net40_positive": lookup[
            "development"
        ].mean_residual_net_12h_40bp
        > 0,
        "validation_residual_net40_positive": lookup[
            "validation"
        ].mean_residual_net_12h_40bp
        > 0,
        "holdout_residual_net40_positive": lookup[
            "holdout"
        ].mean_residual_net_12h_40bp
        > 0,
        "development_raw_net20_positive": lookup[
            "development"
        ].mean_raw_net_12h_20bp
        > 0,
        "validation_raw_net20_positive": lookup[
            "validation"
        ].mean_raw_net_12h_20bp
        > 0,
        "holdout_raw_net20_positive": lookup["holdout"].mean_raw_net_12h_20bp
        > 0,
        "full_raw_net30_positive": lookup["all"].mean_raw_net_12h_30bp > 0,
        "bootstrap_lower_positive": ci_low > 0,
        "label_null_p95": percentile >= 0.95,
        "beats_static": real_mean > static_mean,
        "beats_reversed": real_mean > reversed_mean,
        "beats_delayed": real_mean > delayed_mean,
        "month_share_below_35pct": month_share <= 0.35,
        "worst_period_above_minus40bp": worst_period >= -0.004,
    }
    eligible = all(gates.values())
    return pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "eligible": eligible,
                "verdict": "provisional_forward_shadow_only"
                if eligible
                else "reject_bearish_graph_ml_candidate",
                "full_residual_net40": real_mean,
                "development_residual_net40": lookup[
                    "development"
                ].mean_residual_net_12h_40bp,
                "validation_residual_net40": lookup[
                    "validation"
                ].mean_residual_net_12h_40bp,
                "holdout_residual_net40": lookup[
                    "holdout"
                ].mean_residual_net_12h_40bp,
                "full_raw_net20": lookup["all"].mean_raw_net_12h_20bp,
                "full_raw_net30": lookup["all"].mean_raw_net_12h_30bp,
                "static_residual_net40": static_mean,
                "reversed_residual_net40": reversed_mean,
                "delayed_residual_net40": delayed_mean,
                "label_null_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "max_positive_month_share": month_share,
                "worst_period_mean": worst_period,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        ]
    )


def _write_findings(
    path: Path,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    lines = [
        "# v14.4 Bearish Graph-ML Convergence Findings",
        "",
        f"Verdict: `{audit['verdict'].iloc[0]}`.",
        "",
        "This is an adaptive walk-forward study and cannot independently establish alpha.",
        "",
        "## Audit",
        "",
        audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Eligible pair rows: `{len(panel)}`; walk-forward predicted rows: "
        f"`{len(predictions)}`.",
        "",
        "No PaperLive, leverage, or live-order permission changed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v144_bearish_graph_ml_convergence(
    cfg: V144Config = V144Config(),
) -> dict[str, Path]:
    membership = load_v142_membership(V142Config(membership_path=cfg.membership_path))
    prices = load_v141_price_matrices()
    graph_cfg = V143Config(membership_path=cfg.membership_path)
    edges, contexts = build_v143_graph_and_contexts(prices, membership, graph_cfg)
    pair_panel = build_v144_pair_panel(contexts, edges, cfg)
    if pair_panel.empty:
        raise RuntimeError("v14.4 produced no eligible pair rows")
    predictions = walk_forward_v144_predictions(pair_panel, cfg)
    real = build_v144_model_portfolio(predictions, cfg)
    static = build_v144_static_portfolio(pair_panel, cfg)
    reversed_panel = build_v144_pair_panel(
        contexts, reverse_v143_edges(edges, graph_cfg), cfg
    )
    reversed_predictions = walk_forward_v144_predictions(reversed_panel, cfg)
    reversed_portfolio = build_v144_model_portfolio(reversed_predictions, cfg)
    delayed_panel = build_v144_pair_panel(
        contexts, edges, cfg, state_shift_hours=cfg.delayed_state_hours
    )
    delayed_predictions = walk_forward_v144_predictions(delayed_panel, cfg)
    delayed_portfolio = build_v144_model_portfolio(delayed_predictions, cfg)
    label_nulls = label_null_v144_controls(pair_panel, cfg)
    summary = summarize_v144(real)
    audit = audit_v144(
        real,
        summary,
        static,
        reversed_portfolio,
        delayed_portfolio,
        label_nulls,
        cfg,
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "edges": root / "volatility_receiver_edges.csv",
        "pair_panel": root / "eligible_pair_panel.parquet",
        "predictions": root / "walk_forward_predictions.parquet",
        "portfolio": root / "model_portfolio.parquet",
        "static": root / "static_score_portfolio.parquet",
        "reversed": root / "reversed_graph_portfolio.parquet",
        "delayed": root / "delayed_state_portfolio.parquet",
        "label_nulls": root / "label_permutation_controls.csv",
        "summary": root / "candidate_summary.csv",
        "audit": root / "candidate_audit.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    edges.to_csv(outputs["edges"], index=False)
    pair_panel.to_parquet(outputs["pair_panel"], index=False)
    predictions.to_parquet(outputs["predictions"], index=False)
    real.to_parquet(outputs["portfolio"], index=False)
    static.to_parquet(outputs["static"], index=False)
    reversed_portfolio.to_parquet(outputs["reversed"], index=False)
    delayed_portfolio.to_parquet(outputs["delayed"], index=False)
    label_nulls.to_csv(outputs["label_nulls"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "adaptive_followup": True,
                "candidate": CANDIDATE,
                "pair_rows": len(pair_panel),
                "pair_days": pair_panel["entry_day"].nunique(),
                "predicted_rows": len(predictions),
                "portfolio_observations": len(real),
                "first_deployment_month": (
                    predictions["entry_month"].min() if not predictions.empty else None
                ),
                "label_null_iterations": cfg.label_null_iterations,
                "verdict": audit["verdict"].iloc[0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(cfg.findings_path, audit, summary, pair_panel, predictions)
    return outputs
