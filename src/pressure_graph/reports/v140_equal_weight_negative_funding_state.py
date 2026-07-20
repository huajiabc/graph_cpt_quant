"""Equal-weight all-negative-funding beta-neutral state portfolio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v135_adaptive_negative_funding_breadth import PANEL_PATH
from pressure_graph.reports.v136_severity_weighted_negative_funding import (
    _weighted_components,
    summarize_v136,
)


REPORT_ROOT = Path("reports/v14_0_equal_weight_negative_funding_state")
CANDIDATE = "NF8_ALL_NEGATIVE_EQUAL_BTC_BETA_NEUTRAL"


@dataclass(frozen=True)
class V140Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    minimum_breadth: int = 4
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def load_v140_panel(cfg: V140Config = V140Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    return panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _equal_negative_distribution(
    local: pd.DataFrame,
    cfg: V140Config,
) -> dict[str, float]:
    eligible = local.dropna(subset=["score_7d", "price_return", "btc_beta"])
    symbols = sorted(eligible.loc[eligible["score_7d"].lt(0), "symbol"].astype(str).unique())
    if len(symbols) < cfg.minimum_breadth:
        return {}
    return {symbol: 1.0 / len(symbols) for symbol in symbols}


def build_v140_portfolio(
    panel: pd.DataFrame,
    cfg: V140Config = V140Config(),
) -> pd.DataFrame:
    rows = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        distribution = _equal_negative_distribution(local, cfg)
        if not distribution:
            if previous_weights is not None and rows:
                rows[-1]["realized_turnover"] += sum(
                    abs(weight) for weight in previous_weights.values()
                )
            previous_weights = None
            continue
        weights, components = _weighted_components(local, distribution)
        if not weights:
            continue
        if previous_weights is None:
            turnover = sum(abs(weight) for weight in weights.values())
        else:
            turnover = sum(
                abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in set(weights) | set(previous_weights)
            )
        breadth = len(distribution)
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "eligible_negative_names": breadth,
                "breadth_state": "contracted4to8" if breadth <= 8 else "broad9plus",
                "maximum_base_weight": 1.0 / breadth,
                "selected_symbols": "|".join(distribution),
                "realized_turnover": turnover,
                "_weights": weights,
                **components,
            }
        )
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


def build_v140_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V140Config = V140Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    groups = {
        pd.Timestamp(entry): frame
        for entry, frame in panel.groupby("entry_time", sort=True, observed=True)
    }
    rows = []
    for iteration in range(cfg.null_iterations):
        returns = []
        for row in portfolio.itertuples(index=False):
            local = groups[pd.Timestamp(row.entry_time)]
            usable = local.dropna(subset=["price_return", "btc_beta"])
            symbols = sorted(usable["symbol"].astype(str).unique())
            breadth = int(row.eligible_negative_names)
            selected = list(
                np.asarray(symbols)[rng.choice(len(symbols), size=breadth, replace=False)]
            )
            distribution = {symbol: 1.0 / breadth for symbol in selected}
            _, components = _weighted_components(local, distribution)
            returns.append(
                components["gross_return"] - cfg.one_way_cost * float(row.realized_turnover)
            )
        rows.append(
            {
                "iteration": iteration,
                "null_type": "random_full_universe_same_breadth_observed_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def write_v140_equal_weight_negative_funding_state(
    cfg: V140Config = V140Config(),
) -> dict[str, Path]:
    panel = load_v140_panel(cfg)
    portfolio = build_v140_portfolio(panel, cfg)
    nulls = build_v140_nulls(panel, portfolio, cfg)
    summary = summarize_v136(portfolio, nulls, cfg)
    summary["candidate"] = CANDIDATE
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v140_equal_weight_negative_funding_state_findings_2026_07_15.md"),
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
                "# v14.0 Equal-Weight Negative-Funding State Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The strategy contains no within-state rank. Random controls draw the",
                "same breadth from the full causal cross-section. PaperLive and leverage/",
                "status permissions remain unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
