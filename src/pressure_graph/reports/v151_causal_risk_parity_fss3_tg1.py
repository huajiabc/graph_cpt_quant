"""Causal inverse-volatility allocation between FSS3 and TG1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    _moving_block_means,
)


FSS3_PATH = Path("reports/v14_9_funding_sign_turnover_cap/weekly_portfolio.parquet")
TG1_PATH = Path("reports/v13_2_tg1_forward_temporal_extension/weekly_portfolio.parquet")
REPORT_ROOT = Path("reports/v15_1_causal_risk_parity_fss3_tg1")
FINDINGS_PATH = Path("docs/v151_causal_risk_parity_fss3_tg1_findings_2026_07_16.md")
CANDIDATE = "RP2_CAUSAL_8W_FSS3_TG1"


@dataclass(frozen=True)
class V151Config:
    fss3_path: Path = FSS3_PATH
    tg1_path: Path = TG1_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    volatility_weeks: int = 8
    minimum_tg1_weight: float = 0.25
    maximum_tg1_weight: float = 0.75
    primary_allocation_cost: float = 0.002
    stress_allocation_cost: float = 0.004
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260716


def load_v151_sleeves(cfg: V151Config = V151Config()) -> pd.DataFrame:
    fss3 = pd.read_parquet(cfg.fss3_path)
    tg1 = pd.read_parquet(cfg.tg1_path)
    for frame in (fss3, tg1):
        for column in ("entry_time", "exit_time", "month_start"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    fss3 = fss3[
        [
            "entry_time",
            "exit_time",
            "month_start",
            "period",
            "price_return",
            "funding_return",
            "primary_net_return",
            "stress_net_return",
        ]
    ].rename(
        columns={
            "exit_time": "fss3_exit_time",
            "month_start": "fss3_month_start",
            "period": "fss3_period",
            "price_return": "fss3_price_return",
            "funding_return": "fss3_funding_return",
            "primary_net_return": "fss3_primary_return",
            "stress_net_return": "fss3_stress_return",
        }
    )
    tg1 = tg1[
        [
            "entry_time",
            "exit_time",
            "month_start",
            "period",
            "price_basis_return",
            "funding_spread_return",
            "primary_net_return",
            "stress_net_return",
        ]
    ].rename(
        columns={
            "exit_time": "tg1_exit_time",
            "month_start": "tg1_month_start",
            "period": "tg1_period",
            "price_basis_return": "tg1_price_return",
            "funding_spread_return": "tg1_funding_return",
            "primary_net_return": "tg1_primary_return",
            "stress_net_return": "tg1_stress_return",
        }
    )
    merged = fss3.merge(tg1, on="entry_time", how="inner", validate="one_to_one")
    if not (
        merged["fss3_exit_time"].eq(merged["tg1_exit_time"]).all()
        and merged["fss3_month_start"].eq(merged["tg1_month_start"]).all()
        and merged["fss3_period"].eq(merged["tg1_period"]).all()
    ):
        raise RuntimeError("FSS3 and TG1 calendar labels do not align")
    merged["exit_time"] = merged.pop("fss3_exit_time")
    merged["month_start"] = merged.pop("fss3_month_start")
    merged["period"] = merged.pop("fss3_period")
    return merged.drop(
        columns=["tg1_exit_time", "tg1_month_start", "tg1_period"]
    ).sort_values("entry_time").reset_index(drop=True)


def causal_tg1_weights(
    panel: pd.DataFrame,
    cfg: V151Config = V151Config(),
) -> np.ndarray:
    weights = np.empty(len(panel), dtype=float)
    previous = 0.5
    for index in range(len(panel)):
        if index < cfg.volatility_weeks:
            weight = 0.5
        else:
            history = panel.iloc[index - cfg.volatility_weeks : index]
            tg1_vol = float(history["tg1_primary_return"].std(ddof=1))
            fss3_vol = float(history["fss3_primary_return"].std(ddof=1))
            if tg1_vol > 0 and fss3_vol > 0 and np.isfinite(tg1_vol + fss3_vol):
                weight = fss3_vol / (tg1_vol + fss3_vol)
                weight = float(
                    np.clip(
                        weight,
                        cfg.minimum_tg1_weight,
                        cfg.maximum_tg1_weight,
                    )
                )
            else:
                weight = previous
        weights[index] = weight
        previous = weight
    return weights


def build_v151_portfolio(
    sleeves: pd.DataFrame,
    cfg: V151Config = V151Config(),
) -> pd.DataFrame:
    output = sleeves.copy()
    output["candidate"] = CANDIDATE
    output["tg1_weight"] = causal_tg1_weights(output, cfg)
    output["fss3_weight"] = 1.0 - output["tg1_weight"]
    output["allocation_turnover"] = output["tg1_weight"].diff().abs().fillna(0.0)
    output["primary_allocation_cost"] = (
        cfg.primary_allocation_cost * output["allocation_turnover"]
    )
    output["stress_allocation_cost"] = (
        cfg.stress_allocation_cost * output["allocation_turnover"]
    )
    output["price_return"] = (
        output["tg1_weight"] * output["tg1_price_return"]
        + output["fss3_weight"] * output["fss3_price_return"]
    )
    output["funding_return"] = (
        output["tg1_weight"] * output["tg1_funding_return"]
        + output["fss3_weight"] * output["fss3_funding_return"]
    )
    output["primary_net_return"] = (
        output["tg1_weight"] * output["tg1_primary_return"]
        + output["fss3_weight"] * output["fss3_primary_return"]
        - output["primary_allocation_cost"]
    )
    output["stress_net_return"] = (
        output["tg1_weight"] * output["tg1_stress_return"]
        + output["fss3_weight"] * output["fss3_stress_return"]
        - output["stress_allocation_cost"]
    )
    output["funding_after_primary_allocation_cost"] = (
        output["funding_return"] - output["primary_allocation_cost"]
    )
    output["funding_after_stress_allocation_cost"] = (
        output["funding_return"] - output["stress_allocation_cost"]
    )
    return output


def _additive_max_drawdown(values: pd.Series) -> float:
    curve = values.cumsum()
    return float((curve - curve.cummax().clip(lower=0)).min())


def summarize_v151(
    portfolio: pd.DataFrame,
    cfg: V151Config = V151Config(),
) -> pd.DataFrame:
    values = portfolio["primary_net_return"].to_numpy(dtype=float)
    draws = _moving_block_means(
        values,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + 1),
    )
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    months = portfolio.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    )
    leave_one_month_out = [
        float(
            portfolio.loc[
                portfolio["month_start"].ne(month), "primary_net_return"
            ].mean()
        )
        for month in months.index
    ]
    counts = portfolio["period"].value_counts()
    correlation = float(
        portfolio[["tg1_primary_return", "fss3_primary_return"]].corr().iloc[0, 1]
    )
    combined_drawdown = _additive_max_drawdown(portfolio["primary_net_return"])
    fss3_drawdown = _additive_max_drawdown(portfolio["fss3_primary_return"])
    drawdown_reduction = 1.0 - abs(combined_drawdown) / abs(fss3_drawdown)
    row = {
        "candidate": CANDIDATE,
        "weeks": len(portfolio),
        "months": portfolio["month_start"].nunique(),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "sleeve_correlation": correlation,
        "mean_tg1_bp": portfolio["tg1_primary_return"].mean() * 10_000,
        "mean_fss3_bp": portfolio["fss3_primary_return"].mean() * 10_000,
        "mean_tg1_weight": portfolio["tg1_weight"].mean(),
        "minimum_tg1_weight": portfolio["tg1_weight"].min(),
        "maximum_tg1_weight": portfolio["tg1_weight"].max(),
        "mean_allocation_turnover": portfolio["allocation_turnover"].mean(),
        "mean_primary_allocation_cost_bp": portfolio[
            "primary_allocation_cost"
        ].mean()
        * 10_000,
        "mean_price_bp": portfolio["price_return"].mean() * 10_000,
        "mean_funding_bp": portfolio["funding_return"].mean() * 10_000,
        "mean_primary_net_bp": portfolio["primary_net_return"].mean() * 10_000,
        "mean_stress_net_bp": portfolio["stress_net_return"].mean() * 10_000,
        "mean_funding_after_stress_allocation_cost_bp": portfolio[
            "funding_after_stress_allocation_cost"
        ].mean()
        * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "bootstrap_95_low_bp": ci_low * 10_000,
        "bootstrap_95_high_bp": ci_high * 10_000,
        "positive_month_concentration": concentration,
        "minimum_leave_one_month_out_mean_bp": min(leave_one_month_out) * 10_000,
        "worst_period_bp": periods.min() * 10_000,
        "worst_week_bp": portfolio["primary_net_return"].min() * 10_000,
        "combined_additive_max_drawdown_bp": combined_drawdown * 10_000,
        "fss3_additive_max_drawdown_bp": fss3_drawdown * 10_000,
        "drawdown_reduction_vs_fss3": drawdown_reduction,
    }
    row["promote"] = bool(
        row["weeks"] >= 45
        and row["months"] >= 11
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 10
        and row["mean_tg1_bp"] > 0
        and row["mean_fss3_bp"] > 0
        and abs(row["sleeve_correlation"]) <= 0.50
        and all(
            row[key] > 0
            for key in (
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "mean_stress_net_bp",
                "mean_funding_after_stress_allocation_cost_bp",
                "bootstrap_95_low_bp",
                "minimum_leave_one_month_out_mean_bp",
            )
        )
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and row["drawdown_reduction_vs_fss3"] >= 0.25
    )
    return pd.DataFrame([row])


def write_v151_causal_risk_parity_fss3_tg1(
    cfg: V151Config = V151Config(),
) -> dict[str, Path]:
    sleeves = load_v151_sleeves(cfg)
    portfolio = build_v151_portfolio(sleeves, cfg)
    summary = summarize_v151(portfolio, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_portfolio.parquet",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    portfolio.to_parquet(paths["portfolio"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "weeks": len(portfolio),
                "promoted": promoted,
                "source_sleeves": ["FSS3", "TG1"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_portfolio_shadow" if promoted else "reject_combination"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v15.1 Causal Risk-Parity FSS3 + TG1 Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The allocation architecture is the exact causal v12.9 rule, reused",
                "without inspecting dynamic FSS3/TG1 returns. This is a portfolio-layer",
                "candidate, not a new raw factor. PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
