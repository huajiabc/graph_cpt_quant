"""v2.2D threshold stability audit for the pre-entry meta-router."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v22c_walkforward_policy_simulation import (
    V22CConfig,
    _period_mask,
    _read_or_build_predictions,
    _simulate_policy,
)


REPORT_ROOT = Path("reports/v2_2d_threshold_stability")


@dataclass(frozen=True)
class V22DConfig:
    report_root: Path = REPORT_ROOT
    v22c: V22CConfig = V22CConfig()
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def _baseline_by_period(frame: pd.DataFrame, cfg: V22DConfig) -> dict[str, float]:
    out = {}
    for period in ("search", "validation", "holdout", "full"):
        part = frame[_period_mask(frame, period)].copy()
        _, _, metrics = _simulate_policy(
            part,
            {"kind": "baseline", "mask": pd.Series(False, index=part.index)},
            cfg.v22c,
        )
        out[period] = float(metrics["portfolio_net20"])
    return out


def _row_for_threshold(
    frame: pd.DataFrame,
    *,
    family: str,
    score_col: str,
    threshold: float,
    cfg: V22DConfig,
    baseline: dict[str, float],
) -> dict[str, Any]:
    mask = frame[score_col].ge(threshold)
    _, _, full = _simulate_policy(frame, {"kind": "skip", "mask": mask}, cfg.v22c)
    row: dict[str, Any] = {
        "policy_family": family,
        "score_col": score_col,
        "threshold": threshold,
        **full,
        "delta_vs_baseline_net20": float(full["portfolio_net20"] - baseline["full"]),
    }
    for period in ("search", "validation", "holdout"):
        part = frame[_period_mask(frame, period)].copy()
        period_mask = mask.loc[part.index]
        _, _, metrics = _simulate_policy(part, {"kind": "skip", "mask": period_mask}, cfg.v22c)
        row[f"{period}_portfolio_net20"] = metrics["portfolio_net20"]
        row[f"{period}_delta_vs_baseline_net20"] = float(metrics["portfolio_net20"] - baseline[period])
        row[f"{period}_router_skipped_events"] = metrics["router_affected_events"]
    return row


def _threshold_surface(frame: pd.DataFrame, cfg: V22DConfig) -> pd.DataFrame:
    baseline = _baseline_by_period(frame, cfg)
    rows = []
    families = {
        "logistic_no_trade": "logistic_p_no_trade",
        "stump_no_trade": "stump_p_no_trade",
        "shuffled_logistic_no_trade": "shuffled_logistic_p_no_trade",
    }
    for family, col in families.items():
        for threshold in cfg.thresholds:
            rows.append(_row_for_threshold(frame, family=family, score_col=col, threshold=threshold, cfg=cfg, baseline=baseline))
    return pd.DataFrame(rows)


def _plateau_summary(surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in surface.groupby("policy_family", sort=False):
        ordered = group.sort_values("threshold").copy()
        pass_mask = (
            ordered["delta_vs_baseline_net20"].gt(0)
            & ordered["validation_delta_vs_baseline_net20"].gt(0)
            & ordered["holdout_delta_vs_baseline_net20"].ge(0)
        )
        thresholds = ordered.loc[pass_mask, "threshold"].tolist()
        rows.append(
            {
                "policy_family": family,
                "thresholds_tested": int(len(ordered)),
                "passing_thresholds": int(pass_mask.sum()),
                "passing_threshold_list": ";".join(f"{value:.2f}" for value in thresholds),
                "best_full_delta": float(ordered["delta_vs_baseline_net20"].max()),
                "best_validation_delta": float(ordered["validation_delta_vs_baseline_net20"].max()),
                "best_holdout_delta": float(ordered["holdout_delta_vs_baseline_net20"].max()),
                "median_full_delta": float(ordered["delta_vs_baseline_net20"].median()),
                "delta_range": float(ordered["delta_vs_baseline_net20"].max() - ordered["delta_vs_baseline_net20"].min()),
                "stable_plateau_status": "candidate_plateau" if pass_mask.sum() >= 3 else "no_stable_plateau",
            }
        )
    return pd.DataFrame(rows)


def _notes(root: Path, plateau: pd.DataFrame) -> None:
    lines = [
        "# v2.2D Threshold Stability",
        "",
        "Status: offline threshold audit only. No threshold is promoted.",
        "",
        "## Plateau Summary",
    ]
    for row in plateau.itertuples(index=False):
        lines.append(
            f"- {row.policy_family}: {row.stable_plateau_status}; "
            f"passing={row.passing_thresholds}/{row.thresholds_tested}, "
            f"best_full_delta={row.best_full_delta:.4%}, "
            f"best_validation_delta={row.best_validation_delta:.4%}, "
            f"best_holdout_delta={row.best_holdout_delta:.4%}."
        )
    lines.extend(["", "Decision: threshold robustness alone is insufficient; v2.2E controls must also pass."])
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v22d_threshold_stability(cfg: V22DConfig = V22DConfig()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    frame = _read_or_build_predictions(cfg.v22c)
    surface = _threshold_surface(frame, cfg)
    plateau = _plateau_summary(surface)
    outputs = {
        "threshold_surface": root / "threshold_surface.csv",
        "threshold_plateau_summary": root / "threshold_plateau_summary.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    surface.to_csv(outputs["threshold_surface"], index=False)
    plateau.to_csv(outputs["threshold_plateau_summary"], index=False)
    _notes(root, plateau)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V22DConfig",
    "write_v22d_threshold_stability",
]
