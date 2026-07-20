"""Causal inverse-volatility allocation between TG1 and frozen P2."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


PANEL_PATH = Path("reports/v12_8_tg1_p2_orthogonal_combination/weekly_combination.parquet")
REPORT_ROOT = Path("reports/v12_9_causal_risk_parity_combination")
CANDIDATE = "RP1_CAUSAL_8W_TG1_P2"


@dataclass(frozen=True)
class V129Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    volatility_weeks: int = 8
    minimum_weight: float = 0.25
    maximum_weight: float = 0.75
    allocation_one_way_cost: float = 0.002
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def build_v129_panel(cfg: V129Config = V129Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path).sort_values("entry_time").reset_index(drop=True)
    tg1_weights = []
    previous = 0.5
    for index in range(len(panel)):
        if index < cfg.volatility_weeks:
            weight = 0.5
        else:
            history = panel.iloc[index - cfg.volatility_weeks : index]
            tg1_vol = float(history["tg1_return"].std(ddof=1))
            p2_vol = float(history["p2_return"].std(ddof=1))
            if tg1_vol > 0 and p2_vol > 0 and np.isfinite(tg1_vol + p2_vol):
                weight = p2_vol / (tg1_vol + p2_vol)
                weight = float(np.clip(weight, cfg.minimum_weight, cfg.maximum_weight))
            else:
                weight = previous
        tg1_weights.append(weight)
        previous = weight
    output = panel.copy()
    output["candidate"] = CANDIDATE
    output["tg1_weight"] = tg1_weights
    output["p2_weight"] = 1.0 - output["tg1_weight"]
    output["allocation_turnover"] = output["tg1_weight"].diff().abs().fillna(0.0)
    output["allocation_cost"] = (
        cfg.allocation_one_way_cost * output["allocation_turnover"]
    )
    output["risk_parity_return"] = (
        output["tg1_weight"] * output["tg1_return"]
        + output["p2_weight"] * output["p2_return"]
        - output["allocation_cost"]
    )
    return output


def summarize_v129(
    panel: pd.DataFrame, cfg: V129Config = V129Config()
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    values = panel["risk_parity_return"].to_numpy(dtype=float)
    draws = rng.choice(
        values, size=(cfg.bootstrap_iterations, len(values)), replace=True
    ).mean(axis=1)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = panel.groupby("period", observed=True)["risk_parity_return"].mean()
    months = panel.groupby("month_start", observed=True)["risk_parity_return"].sum()
    positive = months[months.gt(0)]
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
    )
    counts = panel["period"].value_counts()
    correlation = float(panel[["tg1_return", "p2_return"]].corr().iloc[0, 1])
    row = {
        "candidate": CANDIDATE,
        "weeks": len(panel),
        "months": int(panel["month_start"].nunique()),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "sleeve_correlation": correlation,
        "mean_tg1_bp": float(panel["tg1_return"].mean() * 10_000),
        "mean_p2_bp": float(panel["p2_return"].mean() * 10_000),
        "mean_tg1_weight": float(panel["tg1_weight"].mean()),
        "mean_allocation_turnover": float(panel["allocation_turnover"].mean()),
        "mean_allocation_cost_bp": float(panel["allocation_cost"].mean() * 10_000),
        "mean_combined_bp": float(panel["risk_parity_return"].mean() * 10_000),
        "development_combined_bp": float(
            periods.get("development", np.nan) * 10_000
        ),
        "validation_combined_bp": float(
            periods.get("validation", np.nan) * 10_000
        ),
        "holdout_combined_bp": float(periods.get("holdout", np.nan) * 10_000),
        "bootstrap_95_low_bp": float(ci_low * 10_000),
        "bootstrap_95_high_bp": float(ci_high * 10_000),
        "positive_month_concentration": concentration,
        "worst_period_bp": float(periods.min() * 10_000),
    }
    row["promote"] = bool(
        row["weeks"] >= 40
        and row["months"] >= 10
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 8
        and row["mean_tg1_bp"] > 0
        and row["mean_p2_bp"] > 0
        and abs(row["sleeve_correlation"]) <= 0.50
        and all(
            row[key] > 0
            for key in (
                "development_combined_bp",
                "validation_combined_bp",
                "holdout_combined_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
    )
    return pd.DataFrame([row])


def write_v129_causal_risk_parity_combination(
    cfg: V129Config = V129Config(),
) -> dict[str, Path]:
    panel = build_v129_panel(cfg)
    summary = summarize_v129(panel, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_risk_parity.parquet",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v129_causal_risk_parity_combination_findings_2026_07_15.md"
        ),
    }
    panel.to_parquet(paths["panel"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "weeks": len(panel),
                "promoted": [CANDIDATE] if promoted else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_portfolio_shadow" if promoted else "reject_combination"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v12.9 Causal Risk-Parity Combination Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The eight-week causal weighting rule, bounds, and allocation cost were "
                "frozen before inspection. PaperLive was not changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
