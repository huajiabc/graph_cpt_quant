"""Adaptive four-to-nine name negative-funding beta-neutral basket."""

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
from pressure_graph.reports.v134_negative_funding_beta_neutral_rebound import (
    _weights_and_components,
)


PANEL_PATH = Path("reports/v13_4_negative_funding_beta_neutral_rebound/weekly_symbol_panel.parquet")
REPORT_ROOT = Path("reports/v13_5_adaptive_negative_funding_breadth")
CANDIDATE = "NF2_ADAPTIVE4TO9_HOLD18_BTC_BETA_NEUTRAL"


@dataclass(frozen=True)
class V135Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    minimum_breadth: int = 4
    maximum_breadth: int = 9
    hold_rank: int = 18
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def load_v135_panel(cfg: V135Config = V135Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    return panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _select_adaptive_hold_band(
    local: pd.DataFrame,
    previous: list[str],
    cfg: V135Config,
) -> list[str]:
    ranked = local.dropna(subset=["score_7d", "price_return", "btc_beta"])
    ranked = ranked[ranked["score_7d"].lt(0)].sort_values(
        ["score_7d", "symbol"], ascending=[True, True]
    )
    if len(ranked) < cfg.minimum_breadth:
        return []
    target = min(cfg.maximum_breadth, len(ranked))
    ranks = {str(symbol): rank for rank, symbol in enumerate(ranked["symbol"].astype(str), start=1)}
    selected = [
        symbol for symbol in previous if symbol in ranks and ranks[symbol] <= cfg.hold_rank
    ][:target]
    for symbol in ranked["symbol"].astype(str):
        if len(selected) >= target:
            break
        if symbol not in selected:
            selected.append(symbol)
    return selected


def build_v135_portfolio(
    panel: pd.DataFrame,
    cfg: V135Config = V135Config(),
) -> pd.DataFrame:
    rows = []
    previous: list[str] = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        selected = _select_adaptive_hold_band(local, previous, cfg)
        if not selected:
            if previous_weights is not None and rows:
                rows[-1]["realized_turnover"] += sum(
                    abs(weight) for weight in previous_weights.values()
                )
            previous = []
            previous_weights = None
            continue
        weights, components = _weights_and_components(local, selected, cfg)
        if not weights:
            continue
        if previous_weights is None:
            turnover = sum(abs(weight) for weight in weights.values())
        else:
            turnover = sum(
                abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in set(weights) | set(previous_weights)
            )
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "eligible_negative_names": int(local["score_7d"].lt(0).sum()),
                "selected_breadth": len(selected),
                "breadth_state": "full9" if len(selected) == 9 else "contracted4to8",
                "selected_symbols": "|".join(selected),
                "retained_names": len(set(selected) & set(previous)),
                "realized_turnover": turnover,
                "_weights": weights,
                **components,
            }
        )
        previous = selected
        previous_weights = weights
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output.loc[output.index[-1], "realized_turnover"] += sum(
        abs(weight) for weight in output.iloc[-1]["_weights"].values()
    )
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"] - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    return output


def build_v135_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V135Config = V135Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    panel_groups = {
        pd.Timestamp(entry): frame
        for entry, frame in panel.groupby("entry_time", sort=True, observed=True)
    }
    observed = portfolio.set_index("entry_time")
    rows = []
    for iteration in range(cfg.null_iterations):
        returns = []
        for entry, row in observed.iterrows():
            local = panel_groups[pd.Timestamp(entry)]
            eligible = sorted(local.loc[local["score_7d"].lt(0), "symbol"].astype(str).unique())
            breadth = int(row["selected_breadth"])
            selected = list(
                np.asarray(eligible)[rng.choice(len(eligible), size=breadth, replace=False)]
            )
            _, components = _weights_and_components(local, selected, cfg)
            returns.append(
                components["gross_return"] - cfg.one_way_cost * float(row["realized_turnover"])
            )
        rows.append(
            {
                "iteration": iteration,
                "null_type": "random_negative_same_breadth_observed_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def summarize_v135(
    portfolio: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V135Config = V135Config(),
) -> pd.DataFrame:
    values = portfolio["primary_net_return"].to_numpy(dtype=float)
    draws = _moving_block_means(
        values,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + 2),
    )
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    states = portfolio.groupby("breadth_state", observed=True)["primary_net_return"].mean()
    months = portfolio.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
    observed = float(values.mean())
    counts = portfolio["period"].value_counts()
    row = {
        "candidate": CANDIDATE,
        "weeks": len(portfolio),
        "months": int(portfolio["month_start"].nunique()),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "contracted_weeks": int(portfolio["breadth_state"].eq("contracted4to8").sum()),
        "median_selected_breadth": float(portfolio["selected_breadth"].median()),
        "mean_turnover": float(portfolio["realized_turnover"].mean()),
        "mean_price_bp": float(portfolio["price_return"].mean() * 10_000),
        "mean_funding_bp": float(portfolio["funding_return"].mean() * 10_000),
        "mean_gross_bp": float(portfolio["gross_return"].mean() * 10_000),
        "mean_primary_net_bp": observed * 10_000,
        "mean_stress_net_bp": float(portfolio["stress_net_return"].mean() * 10_000),
        "development_primary_net_bp": float(periods.get("development", np.nan) * 10_000),
        "validation_primary_net_bp": float(periods.get("validation", np.nan) * 10_000),
        "holdout_primary_net_bp": float(periods.get("holdout", np.nan) * 10_000),
        "full9_primary_net_bp": float(states.get("full9", np.nan) * 10_000),
        "contracted_primary_net_bp": float(states.get("contracted4to8", np.nan) * 10_000),
        "bootstrap_95_low_bp": float(ci_low * 10_000),
        "bootstrap_95_high_bp": float(ci_high * 10_000),
        "null_percentile": float(100 * nulls["mean_primary_net_return"].le(observed).mean()),
        "positive_month_concentration": concentration,
        "worst_period_bp": float(periods.min() * 10_000),
        "max_abs_residual_btc_beta": float(portfolio["residual_btc_beta"].abs().max()),
    }
    row["promote"] = bool(
        row["weeks"] >= 45
        and row["months"] >= 11
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 10
        and row["mean_turnover"] <= 0.50
        and row["mean_funding_bp"] > 0
        and all(
            row[key] > 0
            for key in (
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "mean_stress_net_bp",
                "full9_primary_net_bp",
                "contracted_primary_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["null_percentile"] >= 90
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and row["max_abs_residual_btc_beta"] <= 1e-12
    )
    return pd.DataFrame([row])


def write_v135_adaptive_negative_funding_breadth(
    cfg: V135Config = V135Config(),
) -> dict[str, Path]:
    panel = load_v135_panel(cfg)
    portfolio = build_v135_portfolio(panel, cfg)
    nulls = build_v135_nulls(panel, portfolio, cfg)
    summary = summarize_v135(portfolio, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v135_adaptive_negative_funding_breadth_findings_2026_07_15.md"),
    }
    portfolio.drop(columns="_weights").to_parquet(paths["portfolio"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "panel_rows": len(panel),
                "weeks": len(portfolio),
                "last_entry": portfolio["entry_time"].max().isoformat(),
                "promoted": [CANDIDATE] if promoted else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_as_tradable_alpha"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v13.5 Adaptive-Breadth Negative-Funding Rebound Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The breadth rule, beta hedge, costs, states, and controls were frozen",
                "before this return was inspected. PaperLive and leverage permissions",
                "remain unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
