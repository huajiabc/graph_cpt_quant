"""Severity-weighted all-negative-funding beta-neutral basket."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    _moving_block_means,
)
from pressure_graph.reports.v135_adaptive_negative_funding_breadth import PANEL_PATH


REPORT_ROOT = Path("reports/v13_6_severity_weighted_negative_funding")
CANDIDATE = "NF3_ALL_NEGATIVE_Q75_SEVERITY_BTC_BETA_NEUTRAL"


@dataclass(frozen=True)
class V136Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    minimum_breadth: int = 4
    severity_cap_quantile: float = 0.75
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def load_v136_panel(cfg: V136Config = V136Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    return panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _severity_distribution(
    local: pd.DataFrame,
    cfg: V136Config,
) -> dict[str, float]:
    eligible = local.dropna(subset=["score_7d", "price_return", "btc_beta"])
    eligible = eligible[eligible["score_7d"].lt(0)].copy()
    if len(eligible) < cfg.minimum_breadth:
        return {}
    severity = -eligible.set_index("symbol")["score_7d"].astype(float)
    cap = float(severity.quantile(cfg.severity_cap_quantile))
    capped = severity.clip(upper=cap)
    if not np.isfinite(capped.sum()) or capped.sum() <= 0:
        return {}
    return {str(symbol): float(value / capped.sum()) for symbol, value in capped.items()}


def _weighted_components(
    local: pd.DataFrame,
    distribution: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    indexed = local.set_index("symbol")
    weighted_beta = float(
        sum(distribution[symbol] * indexed.at[symbol, "btc_beta"] for symbol in distribution)
    )
    if not np.isfinite(weighted_beta) or weighted_beta <= 0:
        return {}, {}
    long_total = 1.0 / (1.0 + weighted_beta)
    btc_short = weighted_beta / (1.0 + weighted_beta)
    weights = {symbol: long_total * base_weight for symbol, base_weight in distribution.items()}
    weights[BTC] = -btc_short
    long_price = float(
        sum(weights[symbol] * indexed.at[symbol, "price_return"] for symbol in distribution)
    )
    btc_price = float(weights[BTC] * indexed.iloc[0]["btc_return"])
    coin_funding = float(
        sum(-weights[symbol] * indexed.at[symbol, "future_funding"] for symbol in distribution)
    )
    btc_funding = float(-weights[BTC] * indexed.iloc[0]["btc_future_funding"])
    residual_beta = float(
        sum(weights[symbol] * indexed.at[symbol, "btc_beta"] for symbol in distribution)
        + weights[BTC]
    )
    return weights, {
        "selected_weighted_beta": weighted_beta,
        "long_notional": long_total,
        "btc_short_notional": btc_short,
        "long_price_return": long_price,
        "btc_hedge_price_return": btc_price,
        "price_return": long_price + btc_price,
        "coin_funding_return": coin_funding,
        "btc_funding_return": btc_funding,
        "funding_return": coin_funding + btc_funding,
        "gross_return": long_price + btc_price + coin_funding + btc_funding,
        "residual_btc_beta": residual_beta,
    }


def build_v136_portfolio(
    panel: pd.DataFrame,
    cfg: V136Config = V136Config(),
) -> pd.DataFrame:
    rows = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        distribution = _severity_distribution(local, cfg)
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
                "maximum_base_weight": max(distribution.values()),
                "selected_symbols": "|".join(sorted(distribution)),
                "realized_turnover": turnover,
                "_base_distribution": distribution,
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


def build_v136_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V136Config = V136Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    panel_groups = {
        pd.Timestamp(entry): frame
        for entry, frame in panel.groupby("entry_time", sort=True, observed=True)
    }
    rows = []
    for iteration in range(cfg.null_iterations):
        returns = []
        for _, row in portfolio.iterrows():
            local = panel_groups[pd.Timestamp(row["entry_time"])]
            observed_distribution = row["_base_distribution"]
            symbols = sorted(observed_distribution)
            values = np.asarray([observed_distribution[symbol] for symbol in symbols])
            shuffled = values[rng.permutation(len(values))]
            distribution = dict(zip(symbols, shuffled, strict=True))
            _, components = _weighted_components(local, distribution)
            returns.append(
                components["gross_return"] - cfg.one_way_cost * float(row["realized_turnover"])
            )
        rows.append(
            {
                "iteration": iteration,
                "null_type": "within_week_severity_weight_permutation_observed_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def summarize_v136(
    portfolio: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V136Config = V136Config(),
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
        "median_negative_breadth": float(portfolio["eligible_negative_names"].median()),
        "mean_maximum_base_weight": float(portfolio["maximum_base_weight"].mean()),
        "mean_turnover": float(portfolio["realized_turnover"].mean()),
        "mean_price_bp": float(portfolio["price_return"].mean() * 10_000),
        "mean_funding_bp": float(portfolio["funding_return"].mean() * 10_000),
        "mean_gross_bp": float(portfolio["gross_return"].mean() * 10_000),
        "mean_primary_net_bp": observed * 10_000,
        "mean_stress_net_bp": float(portfolio["stress_net_return"].mean() * 10_000),
        "development_primary_net_bp": float(periods.get("development", np.nan) * 10_000),
        "validation_primary_net_bp": float(periods.get("validation", np.nan) * 10_000),
        "holdout_primary_net_bp": float(periods.get("holdout", np.nan) * 10_000),
        "contracted_primary_net_bp": float(states.get("contracted4to8", np.nan) * 10_000),
        "broad_primary_net_bp": float(states.get("broad9plus", np.nan) * 10_000),
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
                "contracted_primary_net_bp",
                "broad_primary_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["null_percentile"] >= 90
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and row["max_abs_residual_btc_beta"] <= 1e-12
    )
    return pd.DataFrame([row])


def write_v136_severity_weighted_negative_funding(
    cfg: V136Config = V136Config(),
) -> dict[str, Path]:
    panel = load_v136_panel(cfg)
    portfolio = build_v136_portfolio(panel, cfg)
    nulls = build_v136_nulls(panel, portfolio, cfg)
    summary = summarize_v136(portfolio, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v136_severity_weighted_negative_funding_findings_2026_07_15.md"),
    }
    portfolio.drop(columns=["_base_distribution", "_weights"]).to_parquet(
        paths["portfolio"], index=False
    )
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
                "# v13.6 Severity-Weighted Negative-Funding Basket Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The severity cap, continuous allocation, beta hedge, costs, states,",
                "and permutation controls were frozen before return inspection. No",
                "PaperLive, leverage, or strategy-status permission changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
