"""v2.2G router explainability and deconfounding.

Explains the weak v2.2B pre-entry router signal. This report does not search
for a better selector and does not promote a live/shadow rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v22b_preentry_meta_router import (
    REPORT_ROOT as V22B_ROOT,
    V22BConfig,
    _feature_columns,
    _fit_logistic,
    _num,
    _portfolio_metrics,
    _predict_logistic,
    _train_ready,
    write_v22b_preentry_meta_router,
)


REPORT_ROOT = Path("reports/v2_2g_router_explainability")
V21G_ROOT = Path("reports/v2_1g_meta_router_action_labels")


@dataclass(frozen=True)
class V22GConfig:
    report_root: Path = REPORT_ROOT
    v21g_root: Path = V21G_ROOT
    v22b_root: Path = V22B_ROOT
    v22b: V22BConfig = V22BConfig()
    threshold: float = 0.70


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_inputs(cfg: V22GConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = _read_csv(cfg.v21g_root / "meta_router_event_labels.csv")
    features = _read_csv(cfg.v21g_root / "meta_router_feature_matrix.csv")
    if events.empty or features.empty:
        raise FileNotFoundError(f"Missing v2.1G meta-router inputs under {cfg.v21g_root}")
    predictions_path = cfg.v22b_root / "policy_event_predictions.csv"
    if not predictions_path.exists():
        write_v22b_preentry_meta_router(cfg.v22b)
    predictions = _read_csv(predictions_path)
    if predictions.empty:
        raise FileNotFoundError(f"Missing v2.2B predictions under {cfg.v22b_root}")
    for frame in (events, features, predictions):
        frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
        frame["entry_month"] = frame["entry_month"].astype(str)
    return events, features, predictions


def _period_from_month(month: str) -> str:
    if month < "2026-02":
        return "search"
    if month < "2026-05":
        return "validation"
    return "holdout"


def _fold_coefficients(features: pd.DataFrame, cfg: V22GConfig) -> pd.DataFrame:
    feature_cols = _feature_columns(features)
    rows: list[dict[str, Any]] = []
    for month in sorted(features["entry_month"].dropna().astype(str).unique().tolist()):
        history = features[features["entry_month"].astype(str).lt(month)].copy()
        if not _train_ready(history, cfg.v22b):
            continue
        model = _fit_logistic(history, feature_cols, cfg.v22b)
        weights = np.asarray(model["weights"], dtype=float)
        encoder = model["encoder"]
        pos = 1
        rows.append(
            {
                "eval_month": month,
                "period": _period_from_month(month),
                "feature": "__intercept__",
                "encoded_term": "__intercept__",
                "coefficient": float(weights[0]),
                "abs_coefficient": float(abs(weights[0])),
                "feature_kind": "intercept",
            }
        )
        for col in encoder["numeric_cols"]:
            coef = float(weights[pos])
            rows.append(
                {
                    "eval_month": month,
                    "period": _period_from_month(month),
                    "feature": col,
                    "encoded_term": col,
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                    "feature_kind": "numeric_z",
                }
            )
            pos += 1
        for col in encoder["categorical_cols"]:
            for category in encoder["categories"][col]:
                coef = float(weights[pos])
                rows.append(
                    {
                        "eval_month": month,
                        "period": _period_from_month(month),
                        "feature": col,
                        "encoded_term": f"{col}={category}",
                        "coefficient": coef,
                        "abs_coefficient": abs(coef),
                        "feature_kind": "categorical_onehot",
                    }
                )
                pos += 1
    return pd.DataFrame(rows)


def _coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    data = coefficients[~coefficients["feature"].eq("__intercept__")].copy()
    rows = []
    for feature, group in data.groupby("feature", sort=False):
        coef = pd.to_numeric(group["coefficient"], errors="coerce").dropna()
        if coef.empty:
            continue
        positive = int(coef.gt(0).sum())
        negative = int(coef.lt(0).sum())
        rows.append(
            {
                "feature": feature,
                "fold_terms": int(len(coef)),
                "positive_terms": positive,
                "negative_terms": negative,
                "mean_coefficient": float(coef.mean()),
                "median_coefficient": float(coef.median()),
                "mean_abs_coefficient": float(coef.abs().mean()),
                "sign_stability": float(max(positive, negative) / max(1, len(coef))),
                "coefficient_std": float(coef.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_abs_coefficient", "sign_stability"], ascending=[False, False]
    )


def _score_bucket_summary(predictions: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    data = predictions.copy()
    score = pd.to_numeric(data["logistic_p_no_trade"], errors="coerce")
    try:
        data["router_score_bucket"] = pd.qcut(score.rank(method="first"), buckets, labels=[f"q{i}" for i in range(1, buckets + 1)])
    except ValueError:
        data["router_score_bucket"] = "all"
    rows = []
    for bucket, group in data.groupby("router_score_bucket", sort=False, dropna=False, observed=False):
        net = _num(group, "net20")
        rows.append(
            {
                "router_score_bucket": str(bucket),
                "events": int(len(group)),
                "score_min": float(pd.to_numeric(group["logistic_p_no_trade"], errors="coerce").min()),
                "score_max": float(pd.to_numeric(group["logistic_p_no_trade"], errors="coerce").max()),
                "net20_sum": float(net.sum()),
                "net20_avg": float(net.mean()) if len(group) else np.nan,
                "hit_rate": float(net.gt(0).mean()) if len(group) else np.nan,
                "no_trade_label_rate": float(group["pre_entry_action_label"].eq("no_trade").mean()),
                "cic1_share": float(group["cic_type"].astype(str).eq("CIC1").mean()) if "cic_type" in group else np.nan,
                "market_impulse_density_avg": float(_num(group, "market_impulse_density").mean()),
                "burst_count_avg": float(_num(group, "burst_count_so_far").mean()),
                "peer_count_avg": float(_num(group, "same_timestamp_peer_count").mean()),
                "cluster_density_avg": float(_num(group, "cluster_density").mean()),
                "months": ";".join(sorted(group["entry_month"].astype(str).unique().tolist())),
                "holdout_events": int(group["entry_month"].astype(str).map(_period_from_month).eq("holdout").sum()),
            }
        )
    return pd.DataFrame(rows)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / std


def _low_coimpulse_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions[["trade_key", "entry_month", "period", "net20", "pre_entry_action_label", "logistic_p_no_trade"]].copy()
    market = _zscore(predictions["market_impulse_density"])
    burst = _zscore(predictions["burst_count_so_far"])
    peers = _zscore(predictions["same_timestamp_peer_count"])
    cluster = _zscore(predictions["cluster_density"]).fillna(0.0)
    out["low_coimpulse_score"] = -market.fillna(0.0) - burst.fillna(0.0) - peers.fillna(0.0) - cluster
    out["low_coimpulse_score_norm"] = _zscore(out["low_coimpulse_score"]).fillna(0.0)
    return out


def _as_event_ledger_with_score(events: pd.DataFrame, scores: pd.DataFrame, score_col: str) -> pd.DataFrame:
    score_frame = scores[["trade_key", score_col]].copy()
    out = events.merge(score_frame, on="trade_key", how="left", validate="one_to_one")
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    return out


def _policy_delta(events: pd.DataFrame, score_col: str, threshold: float) -> dict[str, Any]:
    mask = events[score_col].ge(threshold)
    baseline = _portfolio_metrics(events)
    policy = _portfolio_metrics(events, mask)
    out: dict[str, Any] = {
        "selected_trades": policy["selected_trades"],
        "router_skipped_events": policy["router_skipped_events"],
        "portfolio_net20": policy["portfolio_net20"],
        "delta_vs_baseline_net20": policy["portfolio_net20"] - baseline["portfolio_net20"],
        "month_cap35_net20": policy["month_cap35_net20"],
    }
    for period in ("search", "validation", "holdout"):
        period_mask = events["entry_month"].astype(str).map(_period_from_month).eq(period)
        base_period = _portfolio_metrics(events[period_mask])
        policy_period = _portfolio_metrics(events[period_mask], mask[period_mask])
        out[f"{period}_delta_vs_baseline_net20"] = (
            policy_period["portfolio_net20"] - base_period["portfolio_net20"]
        )
    return out


def _month_balanced_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in _feature_columns(out):
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().sum() < 10:
            continue
        out[col] = out.groupby("entry_month")[col].transform(lambda series: _zscore(series).fillna(0.0))
    return out


def _score_from_walkforward(features: pd.DataFrame, events: pd.DataFrame, cfg: V22GConfig) -> pd.DataFrame:
    feature_cols = _feature_columns(features)
    frames = []
    for month in sorted(features["entry_month"].dropna().astype(str).unique().tolist()):
        history = features[features["entry_month"].astype(str).lt(month)].copy()
        current = features[features["entry_month"].astype(str).eq(month)].copy()
        if not _train_ready(history, cfg.v22b):
            current["deconfounded_p_no_trade"] = 0.0
            frames.append(current[["trade_key", "deconfounded_p_no_trade"]])
            continue
        model = _fit_logistic(history, feature_cols, cfg.v22b)
        current["deconfounded_p_no_trade"] = _predict_logistic(model, current)
        frames.append(current[["trade_key", "deconfounded_p_no_trade"]])
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["trade_key", "deconfounded_p_no_trade"])
    return events.merge(scores, on="trade_key", how="left", validate="one_to_one")


def _month_deconfounded_router(events: pd.DataFrame, features: pd.DataFrame, predictions: pd.DataFrame, cfg: V22GConfig) -> pd.DataFrame:
    rows = []
    variants: dict[str, pd.DataFrame] = {
        "raw_logistic_t70": events.merge(
            predictions[["trade_key", "logistic_p_no_trade"]], on="trade_key", how="left", validate="one_to_one"
        ).rename(columns={"logistic_p_no_trade": "router_score"}),
        "month_rank_normalized_features_t70": _score_from_walkforward(
            _month_balanced_feature_frame(features), events, cfg
        ).rename(columns={"deconfounded_p_no_trade": "router_score"}),
    }
    low_scores = _low_coimpulse_scores(predictions)
    low_events = _as_event_ledger_with_score(events, low_scores, "low_coimpulse_score_norm").rename(
        columns={"low_coimpulse_score_norm": "router_score"}
    )
    variants["simple_low_coimpulse_score_t70"] = low_events
    for name, frame in variants.items():
        for threshold in (0.70, 0.80):
            row = {
                "variant": name.replace("_t70", f"_t{int(threshold * 100)}"),
                "score_col": "router_score",
                "threshold": threshold,
                **_policy_delta(frame, "router_score", threshold),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _coefficient_focus(stability: pd.DataFrame) -> pd.DataFrame:
    focus = [
        "market_impulse_density",
        "same_timestamp_peer_count",
        "burst_count_so_far",
        "cluster_density",
        "btc_state",
        "cic_type",
        "beta_strength",
        "local_shock_strength",
        "liquidity_rank",
    ]
    return stability[stability["feature"].isin(focus)].copy()


def _notes(root: Path, stability: pd.DataFrame, buckets: pd.DataFrame, deconfounded: pd.DataFrame) -> None:
    lines = [
        "# v2.2G Router Explainability & Deconfounding",
        "",
        "Status: offline explanation/deconfounding only. No live/shadow selector is promoted.",
        "",
        "## Coefficient Stability",
    ]
    focus = _coefficient_focus(stability).head(12)
    if focus.empty:
        lines.append("- No coefficient stability rows available.")
    else:
        for row in focus.itertuples(index=False):
            lines.append(
                f"- {row.feature}: median_coef={row.median_coefficient:.4f}, "
                f"sign_stability={row.sign_stability:.2f}, mean_abs={row.mean_abs_coefficient:.4f}."
            )
    lines.extend(["", "## Score Buckets"])
    if not buckets.empty:
        for row in buckets.itertuples(index=False):
            lines.append(
                f"- {row.router_score_bucket}: events={row.events}, net20_avg={row.net20_avg:.4%}, "
                f"no_trade_rate={row.no_trade_label_rate:.2f}, "
                f"market={row.market_impulse_density_avg:.3f}, burst={row.burst_count_avg:.2f}, "
                f"peers={row.peer_count_avg:.2f}."
            )
    lines.extend(["", "## Deconfounding"])
    for row in deconfounded.itertuples(index=False):
        lines.append(
            f"- {row.variant}: full_delta={row.delta_vs_baseline_net20:.4%}, "
            f"validation_delta={row.validation_delta_vs_baseline_net20:.4%}, "
            f"holdout_delta={row.holdout_delta_vs_baseline_net20:.4%}, skipped={row.router_skipped_events}."
        )
    lines.extend(
        [
            "",
            "Decision: use this report to decide whether v2.2H should test a transparent conservative veto.",
            "Do not add live/shadow action from v2.2G directly.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v22g_router_explainability(cfg: V22GConfig = V22GConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    events, features, predictions = _load_inputs(cfg)
    coefficients = _fold_coefficients(features, cfg)
    stability = _coefficient_stability(coefficients)
    buckets = _score_bucket_summary(predictions)
    low_scores = _low_coimpulse_scores(predictions)
    deconfounded = _month_deconfounded_router(events, features, predictions, cfg)
    outputs = {
        "logistic_coefficients_by_fold": root / "logistic_coefficients_by_fold.csv",
        "coefficient_stability_summary": root / "coefficient_stability_summary.csv",
        "router_score_bucket_summary": root / "router_score_bucket_summary.csv",
        "low_coimpulse_score_detail": root / "low_coimpulse_score_detail.csv",
        "month_deconfounded_router": root / "month_deconfounded_router.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    coefficients.to_csv(outputs["logistic_coefficients_by_fold"], index=False)
    stability.to_csv(outputs["coefficient_stability_summary"], index=False)
    buckets.to_csv(outputs["router_score_bucket_summary"], index=False)
    low_scores.to_csv(outputs["low_coimpulse_score_detail"], index=False)
    deconfounded.to_csv(outputs["month_deconfounded_router"], index=False)
    _notes(root, stability, buckets, deconfounded)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V22GConfig",
    "write_v22g_router_explainability",
]
