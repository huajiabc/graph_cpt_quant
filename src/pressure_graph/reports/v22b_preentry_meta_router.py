"""v2.2B offline pre-entry meta-router.

Fits simple prior-month-only pre-entry routers over the v2.1G as-of feature
matrix and evaluates them through the existing B4 portfolio architecture. This
is an offline research report only; no live/shadow selector is promoted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import BENCHMARKS, _simulate_portfolio
from pressure_graph.reports.v22a_meta_router_dataset_audit import PRIMARY_FEATURE_BLOCKLIST


REPORT_ROOT = Path("reports/v2_2b_preentry_meta_router")
V21G_ROOT = Path("reports/v2_1g_meta_router_action_labels")
BASE_SPEC = BENCHMARKS["B4_P2_max8_ProtectA_cap2_O6"]
THRESHOLDS = (0.60, 0.70, 0.80)
SEED = 20260614


@dataclass(frozen=True)
class V22BConfig:
    report_root: Path = REPORT_ROOT
    v21g_root: Path = V21G_ROOT
    seed: int = SEED
    min_train_months: int = 3
    min_train_events: int = 60
    l2: float = 0.25
    logistic_steps: int = 600
    logistic_lr: float = 0.08
    random_permutations: int = 100


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _month_cap(values: pd.Series, cap: float = 0.35) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    total = float(numeric.sum())
    cap_value = total * cap if total > 0 else 0.0
    capped = [min(value, cap_value) if value > 0 and cap_value > 0 else value for value in numeric]
    return float(np.sum(capped))


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {
        "trade_key",
        "symbol",
        "candidate",
        "entry_time",
        "entry_month",
        "period",
        "meta_router_training_split",
    }
    blocked_prefixes = ("utility_",)
    blocked_exact = {
        "net20",
        "pre_entry_action_label",
        "post_entry_checkpoint_label",
        "protect_a_label",
        "capacity_overflow_label",
        "checkpoint_exit_delta_vs_keep",
        "protect_keep_delta_vs_cp60",
        *PRIMARY_FEATURE_BLOCKLIST,
    }
    cols = []
    for col in frame.columns:
        if col in metadata or col in blocked_exact or col.startswith(blocked_prefixes):
            continue
        if frame[col].notna().sum() == 0:
            continue
        cols.append(col)
    return cols


def _encode_fit(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in feature_cols:
        numeric = pd.to_numeric(train[col], errors="coerce")
        if numeric.notna().sum() >= max(5, int(len(train) * 0.2)):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    means = {col: float(pd.to_numeric(train[col], errors="coerce").mean()) for col in numeric_cols}
    stds = {}
    for col in numeric_cols:
        std = float(pd.to_numeric(train[col], errors="coerce").std(ddof=0))
        stds[col] = std if std > 1e-12 else 1.0
    categories = {
        col: sorted(train[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
    }
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "means": means,
        "stds": stds,
        "categories": categories,
    }


def _encode_transform(frame: pd.DataFrame, encoder: dict[str, Any]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for col in encoder["numeric_cols"]:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(encoder["means"][col])
        parts.append(((values - encoder["means"][col]) / encoder["stds"][col]).to_numpy(dtype=float)[:, None])
    for col in encoder["categorical_cols"]:
        text = frame[col].astype(str).fillna("")
        for category in encoder["categories"][col]:
            parts.append(text.eq(category).astype(float).to_numpy()[:, None])
    if not parts:
        return np.zeros((len(frame), 0), dtype=float)
    return np.concatenate(parts, axis=1)


def _fit_logistic(train: pd.DataFrame, feature_cols: list[str], cfg: V22BConfig, *, shuffled: bool = False) -> dict[str, Any]:
    labeled = train[train["pre_entry_action_label"].isin(["core_trade", "no_trade"])].copy()
    encoder = _encode_fit(labeled, feature_cols)
    x = _encode_transform(labeled, encoder)
    y = labeled["pre_entry_action_label"].eq("no_trade").astype(float).to_numpy()
    if shuffled:
        rng = np.random.default_rng(cfg.seed + len(labeled))
        y = rng.permutation(y)
    x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    weights = np.zeros(x.shape[1], dtype=float)
    for _ in range(cfg.logistic_steps):
        logits = np.clip(x @ weights, -40, 40)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = x.T @ (pred - y) / max(1, len(y))
        grad[1:] += cfg.l2 * weights[1:] / max(1, len(y))
        weights -= cfg.logistic_lr * grad
    return {"encoder": encoder, "weights": weights, "model_status": "fit"}


def _predict_logistic(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    x = _encode_transform(frame, model["encoder"])
    x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    logits = np.clip(x @ model["weights"], -40, 40)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_stump(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    labeled = train[train["pre_entry_action_label"].isin(["core_trade", "no_trade"])].copy()
    y = labeled["pre_entry_action_label"].eq("no_trade").astype(float)
    base = float(y.mean()) if len(y) else 0.0
    best: dict[str, Any] = {
        "feature": "",
        "operator": "all",
        "threshold": np.nan,
        "left_prob": base,
        "right_prob": base,
        "score": -np.inf,
        "model_status": "fallback_base_rate",
    }
    for col in feature_cols:
        numeric = pd.to_numeric(labeled[col], errors="coerce")
        if numeric.notna().any():
            numeric = numeric.astype(float)
        if numeric.notna().sum() >= 20:
            for threshold in numeric.quantile([0.25, 0.5, 0.75]).dropna().unique():
                mask = numeric.le(float(threshold))
                if mask.sum() < 5 or (~mask).sum() < 5:
                    continue
                left = float(y[mask].mean())
                right = float(y[~mask].mean())
                pred = np.where(mask, left, right)
                score = float(((pred - base) ** 2).mean())
                if score > best["score"]:
                    best = {
                        "feature": col,
                        "operator": "<=",
                        "threshold": float(threshold),
                        "left_prob": left,
                        "right_prob": right,
                        "score": score,
                        "model_status": "fit",
                    }
        else:
            text = labeled[col].astype(str)
            for value in text.value_counts().head(6).index:
                mask = text.eq(str(value))
                if mask.sum() < 5 or (~mask).sum() < 5:
                    continue
                left = float(y[mask].mean())
                right = float(y[~mask].mean())
                pred = np.where(mask, left, right)
                score = float(((pred - base) ** 2).mean())
                if score > best["score"]:
                    best = {
                        "feature": col,
                        "operator": "==",
                        "threshold": str(value),
                        "left_prob": left,
                        "right_prob": right,
                        "score": score,
                        "model_status": "fit",
                    }
    return best


def _predict_stump(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if model.get("model_status") != "fit" or not model.get("feature"):
        return np.full(len(frame), float(model.get("left_prob", 0.0)))
    col = str(model["feature"])
    if model["operator"] == "<=":
        mask = pd.to_numeric(frame[col], errors="coerce").le(float(model["threshold"]))
    else:
        mask = frame[col].astype(str).eq(str(model["threshold"]))
    return np.where(mask, float(model["left_prob"]), float(model["right_prob"]))


def _train_ready(history: pd.DataFrame, cfg: V22BConfig) -> bool:
    labeled = history[history["pre_entry_action_label"].isin(["core_trade", "no_trade"])]
    if len(labeled) < cfg.min_train_events:
        return False
    if labeled["entry_month"].nunique() < cfg.min_train_months:
        return False
    counts = labeled["pre_entry_action_label"].value_counts()
    return bool(counts.get("core_trade", 0) >= 20 and counts.get("no_trade", 0) >= 20)


def _walkforward_predictions(frame: pd.DataFrame, feature_cols: list[str], cfg: V22BConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    months = sorted(frame["entry_month"].dropna().astype(str).unique().tolist())
    for month in months:
        history = frame[frame["entry_month"].astype(str).lt(month)].copy()
        current = frame[frame["entry_month"].astype(str).eq(month)].copy()
        current["train_months"] = history["entry_month"].nunique()
        current["train_events"] = int(len(history))
        if not _train_ready(history, cfg):
            current["logistic_p_no_trade"] = 0.0
            current["stump_p_no_trade"] = 0.0
            current["shuffled_logistic_p_no_trade"] = 0.0
            current["model_status"] = "insufficient_history"
            current["stump_rule"] = ""
            rows.append(current)
            continue
        logistic = _fit_logistic(history, feature_cols, cfg)
        shuffled = _fit_logistic(history, feature_cols, cfg, shuffled=True)
        stump = _fit_stump(history, feature_cols)
        current["logistic_p_no_trade"] = _predict_logistic(logistic, current)
        current["shuffled_logistic_p_no_trade"] = _predict_logistic(shuffled, current)
        current["stump_p_no_trade"] = _predict_stump(stump, current)
        current["model_status"] = "fit"
        current["stump_rule"] = (
            f"{stump.get('feature')} {stump.get('operator')} {stump.get('threshold')} "
            f"=> {stump.get('left_prob'):.3f}/{stump.get('right_prob'):.3f}"
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _period_from_month(month: str) -> str:
    if month < "2026-02":
        return "search"
    if month < "2026-05":
        return "validation"
    return "holdout"


def _portfolio_metrics(data: pd.DataFrame, skipped_by_router: pd.Series | None = None) -> dict[str, Any]:
    if skipped_by_router is not None:
        sample = data[~skipped_by_router.fillna(False)].copy()
        router_skipped_events = int(skipped_by_router.fillna(False).sum())
    else:
        sample = data.copy()
        router_skipped_events = 0
    ledger, skipped = _simulate_portfolio(sample, BASE_SPEC)
    denom = int(BASE_SPEC.max_positions)
    if ledger.empty:
        return {
            "selected_trades": 0,
            "capacity_skipped_trades": int(len(skipped)),
            "router_skipped_events": router_skipped_events,
            "portfolio_net20": 0.0,
            "month_cap35_net20": np.nan,
            "worst_month_net20": np.nan,
            "worst_burst_net20": np.nan,
            "max_month_contribution": np.nan,
        }
    ledger = ledger.copy()
    weighted = _num(ledger, "weighted_return")
    month_returns = weighted.groupby(ledger["entry_month"].astype(str), sort=False).sum() / denom
    burst_returns = weighted.groupby(ledger["burst_id"].astype(str), sort=False).sum() / denom
    positive_months = month_returns[month_returns.gt(0)]
    max_month_contribution = (
        float(positive_months.max() / positive_months.sum()) if not positive_months.empty and positive_months.sum() > 0 else np.nan
    )
    return {
        "selected_trades": int(len(ledger)),
        "capacity_skipped_trades": int(len(skipped)),
        "router_skipped_events": router_skipped_events,
        "portfolio_net20": float(weighted.sum() / denom),
        "month_cap35_net20": _month_cap(month_returns),
        "worst_month_net20": float(month_returns.min()) if len(month_returns) else np.nan,
        "worst_burst_net20": float(burst_returns.min()) if len(burst_returns) else np.nan,
        "max_month_contribution": max_month_contribution,
    }


def _policy_masks(predictions: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {"baseline_B4": pd.Series(False, index=predictions.index)}
    for threshold in THRESHOLDS:
        suffix = int(threshold * 100)
        masks[f"logistic_no_trade_t{suffix}"] = predictions["logistic_p_no_trade"].ge(threshold)
        masks[f"stump_no_trade_t{suffix}"] = predictions["stump_p_no_trade"].ge(threshold)
        masks[f"shuffled_logistic_no_trade_t{suffix}"] = predictions["shuffled_logistic_p_no_trade"].ge(threshold)
    return masks


def _evaluate_policies(predictions: pd.DataFrame, masks: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    baseline_by_month: dict[str, float] = {}
    for month, month_frame in predictions.groupby("entry_month", sort=True):
        baseline_metrics = _portfolio_metrics(month_frame)
        baseline_by_month[str(month)] = float(baseline_metrics["portfolio_net20"])
    for policy_id, mask in masks.items():
        full_metrics = _portfolio_metrics(predictions, mask)
        row = {"policy_id": policy_id, **full_metrics}
        monthly_net = []
        for month, month_frame in predictions.groupby("entry_month", sort=True):
            month_mask = mask.loc[month_frame.index]
            metrics = _portfolio_metrics(month_frame, month_mask)
            period = _period_from_month(str(month))
            monthly_net.append(metrics["portfolio_net20"])
            monthly_rows.append(
                {
                    "policy_id": policy_id,
                    "entry_month": month,
                    "period": period,
                    **metrics,
                    "baseline_month_net20": baseline_by_month[str(month)],
                    "delta_vs_baseline_net20": float(metrics["portfolio_net20"] - baseline_by_month[str(month)]),
                }
            )
        monthly_df = pd.DataFrame([r for r in monthly_rows if r["policy_id"] == policy_id])
        for period in ("search", "validation", "holdout"):
            part = monthly_df[monthly_df["period"].eq(period)]
            row[f"{period}_portfolio_net20"] = float(part["portfolio_net20"].sum()) if not part.empty else 0.0
            row[f"{period}_delta_vs_baseline_net20"] = float(part["delta_vs_baseline_net20"].sum()) if not part.empty else 0.0
        row["delta_vs_baseline_net20"] = float(
            row["portfolio_net20"] - sum(baseline_by_month.values())
        )
        row["months_improved"] = int(monthly_df["delta_vs_baseline_net20"].gt(0).sum()) if not monthly_df.empty else 0
        row["months_worse"] = int(monthly_df["delta_vs_baseline_net20"].lt(0).sum()) if not monthly_df.empty else 0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["validation_delta_vs_baseline_net20", "holdout_delta_vs_baseline_net20", "delta_vs_baseline_net20"],
        ascending=[False, False, False],
    )
    return summary.reset_index(drop=True), pd.DataFrame(monthly_rows)


def _random_controls(predictions: pd.DataFrame, reference_mask: pd.Series, cfg: V22BConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    months = sorted(predictions["entry_month"].dropna().astype(str).unique().tolist())
    baseline = _portfolio_metrics(predictions)["portfolio_net20"]
    for perm in range(cfg.random_permutations):
        mask = pd.Series(False, index=predictions.index)
        for month in months:
            idx = predictions[predictions["entry_month"].astype(str).eq(month)].index.to_numpy()
            skip_count = int(reference_mask.loc[idx].sum())
            if skip_count <= 0:
                continue
            chosen = rng.choice(idx, size=min(skip_count, len(idx)), replace=False)
            mask.loc[chosen] = True
        metrics = _portfolio_metrics(predictions, mask)
        rows.append(
            {
                "control_id": "random_skip_match_logistic_t70",
                "permutation": perm,
                "portfolio_net20": metrics["portfolio_net20"],
                "delta_vs_baseline_net20": float(metrics["portfolio_net20"] - baseline),
                "router_skipped_events": metrics["router_skipped_events"],
            }
        )
    return pd.DataFrame(rows)


def _threshold_stability(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ("logistic_no_trade", "stump_no_trade", "shuffled_logistic_no_trade"):
        part = summary[summary["policy_id"].astype(str).str.startswith(family)].copy()
        if part.empty:
            continue
        rows.append(
            {
                "policy_family": family,
                "thresholds": int(len(part)),
                "positive_full_delta_thresholds": int(part["delta_vs_baseline_net20"].gt(0).sum()),
                "positive_validation_delta_thresholds": int(part["validation_delta_vs_baseline_net20"].gt(0).sum()),
                "positive_holdout_delta_thresholds": int(part["holdout_delta_vs_baseline_net20"].gt(0).sum()),
                "best_full_delta": float(part["delta_vs_baseline_net20"].max()),
                "best_validation_delta": float(part["validation_delta_vs_baseline_net20"].max()),
                "best_holdout_delta": float(part["holdout_delta_vs_baseline_net20"].max()),
            }
        )
    return pd.DataFrame(rows)


def _notes(root: Path, summary: pd.DataFrame, random_controls: pd.DataFrame) -> None:
    lines = [
        "# v2.2B Pre-entry Meta-router",
        "",
        "Status: offline walk-forward research only. No selector, shadow, paper-live, or real-live rule is promoted.",
        "",
        "## Policy Summary",
    ]
    for row in summary.head(8).itertuples(index=False):
        random_pct = getattr(row, "random_skip_match_t70_percentile", np.nan)
        lines.append(
            f"- {row.policy_id}: full_delta={row.delta_vs_baseline_net20:.4%}, "
            f"validation_delta={row.validation_delta_vs_baseline_net20:.4%}, "
            f"holdout_delta={row.holdout_delta_vs_baseline_net20:.4%}, "
            f"router_skipped={row.router_skipped_events}, "
            f"random_percentile={random_pct:.2f}."
        )
    if not random_controls.empty:
        q = random_controls["delta_vs_baseline_net20"].quantile([0.5, 0.75, 0.9]).to_dict()
        lines.extend(
            [
                "",
                "## Random Control",
                f"- random_skip_match_logistic_t70 delta median={q.get(0.5, np.nan):.4%}, "
                f"p75={q.get(0.75, np.nan):.4%}, p90={q.get(0.9, np.nan):.4%}.",
            ]
        )
    candidates = summary[
        ~summary["policy_id"].eq("baseline_B4")
        & summary["delta_vs_baseline_net20"].gt(0)
        & summary["validation_delta_vs_baseline_net20"].gt(0)
        & summary["holdout_delta_vs_baseline_net20"].ge(0)
        & summary.get("random_skip_match_t70_percentile", pd.Series(0.0, index=summary.index)).ge(0.75)
        & ~summary["policy_id"].str.contains("shuffled", regex=False)
    ]
    lines.extend(["", "## Decision"])
    if candidates.empty:
        lines.append(
            "- No pre-entry meta-router passed validation + holdout + random-p75 guardrails. "
            "Do not promote to shadow."
        )
    else:
        for row in candidates.itertuples(index=False):
            lines.append(
                f"- Offline candidate for further audit: {row.policy_id}, "
                f"full_delta={row.delta_vs_baseline_net20:.4%}."
            )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v22b_preentry_meta_router(cfg: V22BConfig = V22BConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    labels = _read_csv(cfg.v21g_root / "meta_router_event_labels.csv")
    features = _read_csv(cfg.v21g_root / "meta_router_feature_matrix.csv")
    if labels.empty:
        raise FileNotFoundError(f"Missing or empty v2.1G event labels under {cfg.v21g_root}")
    if features.empty:
        raise FileNotFoundError(f"Missing or empty v2.1G feature matrix under {cfg.v21g_root}")
    labels = labels.copy()
    features = features.copy()
    labels["entry_time"] = pd.to_datetime(labels["entry_time"], utc=True, errors="coerce")
    labels["entry_month"] = labels["entry_month"].astype(str)
    features["entry_time"] = pd.to_datetime(features["entry_time"], utc=True, errors="coerce")
    features["entry_month"] = features["entry_month"].astype(str)
    feature_cols = _feature_columns(features)
    predictions = _walkforward_predictions(features, feature_cols, cfg)
    prediction_cols = [
        "trade_key",
        "logistic_p_no_trade",
        "stump_p_no_trade",
        "shuffled_logistic_p_no_trade",
        "model_status",
        "stump_rule",
        "train_months",
        "train_events",
    ]
    scored = labels.merge(predictions[prediction_cols], on="trade_key", how="left", validate="one_to_one")
    for col in ("logistic_p_no_trade", "stump_p_no_trade", "shuffled_logistic_p_no_trade"):
        scored[col] = pd.to_numeric(scored[col], errors="coerce").fillna(0.0)
    scored["model_status"] = scored["model_status"].fillna("missing_prediction")
    masks = _policy_masks(scored)
    summary, monthly = _evaluate_policies(scored, masks)
    random_controls = _random_controls(scored, masks["logistic_no_trade_t70"], cfg)
    if not random_controls.empty:
        control_delta = pd.to_numeric(random_controls["delta_vs_baseline_net20"], errors="coerce").dropna()
        summary["random_skip_match_t70_percentile"] = [
            float(control_delta.le(delta).mean()) if len(control_delta) else np.nan
            for delta in pd.to_numeric(summary["delta_vs_baseline_net20"], errors="coerce")
        ]
    else:
        summary["random_skip_match_t70_percentile"] = np.nan
    threshold_stability = _threshold_stability(summary)

    outputs = {
        "policy_event_predictions": root / "policy_event_predictions.csv",
        "walkforward_policy_summary": root / "walkforward_policy_summary.csv",
        "walkforward_monthly_performance": root / "walkforward_monthly_performance.csv",
        "threshold_stability": root / "threshold_stability.csv",
        "negative_controls": root / "negative_controls.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    scored.to_csv(outputs["policy_event_predictions"], index=False)
    summary.to_csv(outputs["walkforward_policy_summary"], index=False)
    monthly.to_csv(outputs["walkforward_monthly_performance"], index=False)
    threshold_stability.to_csv(outputs["threshold_stability"], index=False)
    random_controls.to_csv(outputs["negative_controls"], index=False)
    _notes(root, summary, random_controls)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V22BConfig",
    "write_v22b_preentry_meta_router",
]
