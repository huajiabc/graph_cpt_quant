"""v2.2E negative controls for the pre-entry meta-router."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v22c_walkforward_policy_simulation import (
    V22CConfig,
    _period_mask,
    _read_or_build_predictions,
    _simulate_policy,
)


REPORT_ROOT = Path("reports/v2_2e_negative_controls")


@dataclass(frozen=True)
class V22EConfig:
    report_root: Path = REPORT_ROOT
    v22c: V22CConfig = V22CConfig()
    seed: int = 20260614
    permutations: int = 100
    threshold: float = 0.70


def _baseline(frame: pd.DataFrame, cfg: V22EConfig) -> dict[str, float]:
    out = {}
    for period in ("search", "validation", "holdout", "full"):
        part = frame[_period_mask(frame, period)].copy()
        _, _, metrics = _simulate_policy(part, {"kind": "baseline", "mask": pd.Series(False, index=part.index)}, cfg.v22c)
        out[period] = float(metrics["portfolio_net20"])
    return out


def _eval_mask(frame: pd.DataFrame, mask: pd.Series, cfg: V22EConfig, baseline: dict[str, float]) -> dict[str, Any]:
    _, _, full = _simulate_policy(frame, {"kind": "skip", "mask": mask}, cfg.v22c)
    row: dict[str, Any] = {
        **full,
        "delta_vs_baseline_net20": float(full["portfolio_net20"] - baseline["full"]),
    }
    for period in ("search", "validation", "holdout"):
        part = frame[_period_mask(frame, period)].copy()
        period_mask = mask.loc[part.index]
        _, _, metrics = _simulate_policy(part, {"kind": "skip", "mask": period_mask}, cfg.v22c)
        row[f"{period}_delta_vs_baseline_net20"] = float(metrics["portfolio_net20"] - baseline[period])
        row[f"{period}_portfolio_net20"] = metrics["portfolio_net20"]
        row[f"{period}_router_skipped_events"] = metrics["router_affected_events"]
    return row


def _month_permuted_mask(frame: pd.DataFrame, score_col: str, threshold: float, rng: np.random.Generator) -> pd.Series:
    shuffled = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("entry_month", sort=False):
        values = group[score_col].to_numpy(dtype=float)
        shuffled.loc[group.index] = rng.permutation(values)
    return shuffled.ge(threshold)


def _controls(frame: pd.DataFrame, cfg: V22EConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _baseline(frame, cfg)
    rng = np.random.default_rng(cfg.seed)
    primary_mask = frame["logistic_p_no_trade"].ge(cfg.threshold)
    primary = _eval_mask(frame, primary_mask, cfg, baseline)
    rows = [
        {
            "control_id": "primary_logistic_t70",
            "control_type": "primary",
            "permutation": -1,
            **primary,
        },
        {
            "control_id": "shuffled_label_model_t70",
            "control_type": "shuffled_label_model",
            "permutation": -1,
            **_eval_mask(frame, frame["shuffled_logistic_p_no_trade"].ge(cfg.threshold), cfg, baseline),
        },
        {
            "control_id": "reverse_skip_low_risk_t70",
            "control_type": "reverse",
            "permutation": -1,
            **_eval_mask(frame, frame["logistic_p_no_trade"].le(1.0 - cfg.threshold), cfg, baseline),
        },
    ]
    skip_by_month = {
        month: int(primary_mask.loc[group.index].sum())
        for month, group in frame.groupby("entry_month", sort=False)
    }
    for idx in range(cfg.permutations):
        random_mask = pd.Series(False, index=frame.index)
        for month, group in frame.groupby("entry_month", sort=False):
            count = skip_by_month[str(month)]
            if count <= 0:
                continue
            chosen = rng.choice(group.index.to_numpy(), size=min(count, len(group)), replace=False)
            random_mask.loc[chosen] = True
        rows.append(
            {
                "control_id": "random_skip_count_matched_t70",
                "control_type": "random_count_matched",
                "permutation": idx,
                **_eval_mask(frame, random_mask, cfg, baseline),
            }
        )
        rows.append(
            {
                "control_id": "month_shuffled_score_t70",
                "control_type": "month_shuffled_score",
                "permutation": idx,
                **_eval_mask(frame, _month_permuted_mask(frame, "logistic_p_no_trade", cfg.threshold, rng), cfg, baseline),
            }
        )
        global_values = rng.permutation(frame["logistic_p_no_trade"].to_numpy(dtype=float))
        rows.append(
            {
                "control_id": "global_shuffled_score_t70",
                "control_type": "global_shuffled_score",
                "permutation": idx,
                **_eval_mask(frame, pd.Series(global_values, index=frame.index).ge(cfg.threshold), cfg, baseline),
            }
        )
    detail = pd.DataFrame(rows)
    summary_rows = []
    primary_delta = float(primary["delta_vs_baseline_net20"])
    for control_type, group in detail.groupby("control_type", sort=False):
        deltas = pd.to_numeric(group["delta_vs_baseline_net20"], errors="coerce")
        summary_rows.append(
            {
                "control_type": control_type,
                "runs": int(len(group)),
                "delta_mean": float(deltas.mean()),
                "delta_median": float(deltas.median()),
                "delta_p75": float(deltas.quantile(0.75)),
                "delta_p90": float(deltas.quantile(0.90)),
                "delta_max": float(deltas.max()),
                "primary_percentile_vs_control": float(deltas.le(primary_delta).mean()),
                "validation_positive_runs": int(pd.to_numeric(group["validation_delta_vs_baseline_net20"], errors="coerce").gt(0).sum()),
                "holdout_positive_runs": int(pd.to_numeric(group["holdout_delta_vs_baseline_net20"], errors="coerce").gt(0).sum()),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _notes(root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# v2.2E Negative Controls",
        "",
        "Status: offline controls only. No selector is promoted.",
        "",
        "## Control Summary",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.control_type}: runs={row.runs}, median={row.delta_median:.4%}, "
            f"p75={row.delta_p75:.4%}, p90={row.delta_p90:.4%}, "
            f"primary_percentile={row.primary_percentile_vs_control:.2f}."
        )
    random = summary[summary["control_type"].eq("random_count_matched")]
    passed = bool(not random.empty and float(random.iloc[0]["primary_percentile_vs_control"]) >= 0.75)
    lines.extend(["", "## Decision"])
    if passed:
        lines.append("- Primary router clears random-count p75, but must still pass v2.2F synthesis.")
    else:
        lines.append("- Primary router does not clear random-count p75. Do not promote to shadow.")
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v22e_negative_controls(cfg: V22EConfig = V22EConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    frame = _read_or_build_predictions(cfg.v22c)
    detail, summary = _controls(frame, cfg)
    outputs = {
        "negative_control_detail": root / "negative_control_detail.csv",
        "negative_control_summary": root / "negative_control_summary.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    detail.to_csv(outputs["negative_control_detail"], index=False)
    summary.to_csv(outputs["negative_control_summary"], index=False)
    _notes(root, summary)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V22EConfig",
    "write_v22e_negative_controls",
]
