"""Fixed 80/20 FSS3 core plus TG1 satellite portfolio."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    _moving_block_means,
)
from pressure_graph.reports.v151_causal_risk_parity_fss3_tg1 import (
    FSS3_PATH,
    TG1_PATH,
    _additive_max_drawdown,
    load_v151_sleeves,
)


REPORT_ROOT = Path("reports/v16_5_fixed_core_satellite_fss3_tg1")
FINDINGS_PATH = Path("docs/v165_fixed_core_satellite_fss3_tg1_findings_2026_07_16.md")
CANDIDATE = "CM2_FIXED_80_FSS3_20_TG1"


@dataclass(frozen=True)
class V165Config:
    fss3_path: Path = FSS3_PATH
    tg1_path: Path = TG1_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    fss3_weight: float = 0.80
    tg1_weight: float = 0.20
    bootstrap_iterations: int = 10_000
    bootstrap_block_weeks: int = 4
    seed: int = 20260728


def load_v165_sleeves(cfg: V165Config = V165Config()) -> pd.DataFrame:
    from pressure_graph.reports.v151_causal_risk_parity_fss3_tg1 import V151Config

    return load_v151_sleeves(
        V151Config(fss3_path=cfg.fss3_path, tg1_path=cfg.tg1_path)
    )


def build_v165_portfolio(
    sleeves: pd.DataFrame,
    cfg: V165Config = V165Config(),
) -> pd.DataFrame:
    if not np.isclose(cfg.fss3_weight + cfg.tg1_weight, 1.0):
        raise ValueError("fixed sleeve weights must sum to one")
    output = sleeves.copy()
    output["candidate"] = CANDIDATE
    output["fss3_weight"] = cfg.fss3_weight
    output["tg1_weight"] = cfg.tg1_weight
    output["price_return"] = (
        cfg.fss3_weight * output["fss3_price_return"]
        + cfg.tg1_weight * output["tg1_price_return"]
    )
    output["funding_return"] = (
        cfg.fss3_weight * output["fss3_funding_return"]
        + cfg.tg1_weight * output["tg1_funding_return"]
    )
    output["primary_net_return"] = (
        cfg.fss3_weight * output["fss3_primary_return"]
        + cfg.tg1_weight * output["tg1_primary_return"]
    )
    output["stress_net_return"] = (
        cfg.fss3_weight * output["fss3_stress_return"]
        + cfg.tg1_weight * output["tg1_stress_return"]
    )
    return output


def _downside_semideviation(values: pd.Series) -> float:
    downside = np.minimum(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float), 0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def summarize_v165(
    portfolio: pd.DataFrame,
    cfg: V165Config = V165Config(),
) -> pd.DataFrame:
    draws = _moving_block_means(
        portfolio["primary_net_return"].to_numpy(dtype=float),
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed),
    )
    bootstrap_low, bootstrap_high = np.quantile(draws, [0.025, 0.975])
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    counts = portfolio["period"].value_counts()
    months = portfolio.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    leave_one_month_out = [
        float(
            portfolio.loc[
                portfolio["month_start"].ne(month), "primary_net_return"
            ].mean()
        )
        for month in months.index
    ]
    correlation = float(
        portfolio[["fss3_primary_return", "tg1_primary_return"]].corr().iloc[0, 1]
    )
    combined_drawdown = _additive_max_drawdown(portfolio["primary_net_return"])
    fss3_drawdown = _additive_max_drawdown(portfolio["fss3_primary_return"])
    drawdown_reduction = 1.0 - abs(combined_drawdown) / abs(fss3_drawdown)
    combined_semideviation = _downside_semideviation(portfolio["primary_net_return"])
    fss3_semideviation = _downside_semideviation(portfolio["fss3_primary_return"])
    fss3_mean = float(portfolio["fss3_primary_return"].mean())
    combined_mean = float(portfolio["primary_net_return"].mean())
    row = {
        "candidate": CANDIDATE,
        "weeks": len(portfolio),
        "months": portfolio["month_start"].nunique(),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "sleeve_correlation": correlation,
        "fss3_weight": cfg.fss3_weight,
        "tg1_weight": cfg.tg1_weight,
        "max_weight_drift": max(
            float((portfolio["fss3_weight"] - cfg.fss3_weight).abs().max()),
            float((portfolio["tg1_weight"] - cfg.tg1_weight).abs().max()),
        ),
        "mean_fss3_bp": fss3_mean * 10_000,
        "mean_tg1_bp": portfolio["tg1_primary_return"].mean() * 10_000,
        "mean_price_bp": portfolio["price_return"].mean() * 10_000,
        "mean_funding_bp": portfolio["funding_return"].mean() * 10_000,
        "mean_primary_net_bp": combined_mean * 10_000,
        "mean_stress_net_bp": portfolio["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "bootstrap_95_low_bp": bootstrap_low * 10_000,
        "bootstrap_95_high_bp": bootstrap_high * 10_000,
        "positive_month_concentration": concentration,
        "minimum_leave_one_month_out_mean_bp": min(leave_one_month_out) * 10_000,
        "mean_retention_vs_fss3": combined_mean / fss3_mean,
        "combined_additive_max_drawdown_bp": combined_drawdown * 10_000,
        "fss3_additive_max_drawdown_bp": fss3_drawdown * 10_000,
        "drawdown_reduction_vs_fss3": drawdown_reduction,
        "combined_downside_semideviation_bp": combined_semideviation * 10_000,
        "fss3_downside_semideviation_bp": fss3_semideviation * 10_000,
        "downside_semideviation_reduction_vs_fss3": 1.0
        - combined_semideviation / fss3_semideviation,
        "worst_week_bp": portfolio["primary_net_return"].min() * 10_000,
    }
    row["promote"] = bool(
        row["weeks"] >= 45
        and row["months"] >= 11
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 10
        and row["mean_fss3_bp"] > 0
        and row["mean_tg1_bp"] > 0
        and abs(row["sleeve_correlation"]) <= 0.25
        and all(
            row[key] > 0
            for key in (
                "mean_primary_net_bp",
                "mean_stress_net_bp",
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "bootstrap_95_low_bp",
                "minimum_leave_one_month_out_mean_bp",
            )
        )
        and row["positive_month_concentration"] <= 0.35
        and row["mean_retention_vs_fss3"] >= 0.75
        and row["drawdown_reduction_vs_fss3"] >= 0.15
        and row["combined_downside_semideviation_bp"]
        < row["fss3_downside_semideviation_bp"]
        and row["max_weight_drift"] <= 1e-12
    )
    return pd.DataFrame([row])


def write_v165_fixed_core_satellite_fss3_tg1(
    cfg: V165Config = V165Config(),
) -> dict[str, Path]:
    sleeves = load_v165_sleeves(cfg)
    portfolio = build_v165_portfolio(sleeves, cfg)
    summary = summarize_v165(portfolio, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "sleeves": root / "aligned_sleeves.parquet",
        "portfolio": root / "weekly_portfolio.parquet",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    sleeves.to_parquet(paths["sleeves"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    serialized_config = {
        **asdict(cfg),
        "fss3_path": str(cfg.fss3_path),
        "tg1_path": str(cfg.tg1_path),
        "report_root": str(cfg.report_root),
        "findings_path": str(cfg.findings_path),
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "promoted": promoted,
                "config": serialized_config,
                "scope": "portfolio_layer_forward_shadow_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_portfolio_shadow" if promoted else "reject_combination"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v16.5 Fixed Core-Satellite FSS3 + TG1 Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The 80/20 weights were frozen as a single core-satellite architecture;",
                "no weight grid or dynamic allocation was inspected. This is a portfolio",
                "construction candidate, not a new raw factor. PaperLive and remote state",
                "are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
