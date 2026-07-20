"""Two-week funding-sign confirmation for the v14.7 spread."""
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
from pressure_graph.reports.v147_funding_sign_spread import (
    PANEL_PATH,
    REPORT_ROOT as V147_REPORT_ROOT,
    V147Config,
    _turnover_with_terminal_close,
    beta_neutral_components,
    load_v147_panel,
)


REPORT_ROOT = Path("reports/v14_8_funding_sign_hysteresis")
FINDINGS_PATH = Path("docs/v148_funding_sign_hysteresis_findings_2026_07_15.md")
CANDIDATE = "FSS2_TWO_WEEK_SIGN_CONFIRMATION"


@dataclass(frozen=True)
class V148Config:
    panel_path: Path = PANEL_PATH
    v147_portfolio_path: Path = V147_REPORT_ROOT / "weekly_portfolios.parquet"
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_side_breadth: int = 4
    confirmation_weeks: int = 2
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def update_sign_states(
    states: dict[str, int],
    opposite_streaks: dict[str, int],
    scores: dict[str, float],
    confirmation_weeks: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Update causal sign states; missing symbols exit immediately."""
    current_symbols = set(scores)
    states = {symbol: side for symbol, side in states.items() if symbol in current_symbols}
    opposite_streaks = {
        symbol: streak
        for symbol, streak in opposite_streaks.items()
        if symbol in current_symbols
    }
    for symbol in sorted(current_symbols):
        score = float(scores[symbol])
        desired = -1 if score < 0 else 1 if score > 0 else 0
        if symbol not in states:
            if desired != 0:
                states[symbol] = desired
                opposite_streaks[symbol] = 0
            continue
        if desired == 0 or desired == states[symbol]:
            opposite_streaks[symbol] = 0
            continue
        streak = opposite_streaks.get(symbol, 0) + 1
        if streak >= confirmation_weeks:
            states[symbol] = desired
            opposite_streaks[symbol] = 0
        else:
            opposite_streaks[symbol] = streak
    return states, opposite_streaks


def build_v148_portfolio(
    panel: pd.DataFrame,
    cfg: V148Config = V148Config(),
) -> pd.DataFrame:
    rows: list[dict] = []
    states: dict[str, int] = {}
    opposite_streaks: dict[str, int] = {}
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        eligible = local.dropna(
            subset=["score_7d", "price_return", "future_funding", "btc_beta"]
        )
        scores = {
            str(row.symbol): float(row.score_7d)
            for row in eligible.itertuples(index=False)
        }
        states, opposite_streaks = update_sign_states(
            states, opposite_streaks, scores, cfg.confirmation_weeks
        )
        long_symbols = sorted(symbol for symbol, side in states.items() if side < 0)
        short_symbols = sorted(symbol for symbol, side in states.items() if side > 0)
        if (
            len(long_symbols) < cfg.minimum_side_breadth
            or len(short_symbols) < cfg.minimum_side_breadth
        ):
            continue
        distribution = {
            symbol: 0.5 / len(long_symbols) for symbol in long_symbols
        }
        distribution.update(
            {symbol: -0.5 / len(short_symbols) for symbol in short_symbols}
        )
        weights, components = beta_neutral_components(local, distribution)
        if not weights:
            continue
        observed_negative_breadth = int(eligible["score_7d"].lt(0).sum())
        disagreements = sum(
            1
            for symbol, side in states.items()
            if scores[symbol] != 0
            and ((scores[symbol] < 0 and side > 0) or (scores[symbol] > 0 and side < 0))
        )
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "negative_breadth": observed_negative_breadth,
                "long_side_breadth": len(long_symbols),
                "short_side_breadth": len(short_symbols),
                "retained_opposite_sign_names": disagreements,
                "breadth_state": (
                    "contracted4to8"
                    if observed_negative_breadth <= 8
                    else "broad9plus"
                ),
                "selected_long_symbols": "|".join(long_symbols),
                "selected_short_symbols": "|".join(short_symbols),
                "realized_turnover": 0.0,
                "_weights": weights,
                **components,
            }
        )
    _turnover_with_terminal_close(rows, CANDIDATE)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"]
        - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    return output


def build_v148_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V148Config = V148Config(),
) -> pd.DataFrame:
    groups = {
        pd.Timestamp(entry): local
        for entry, local in panel.groupby("entry_time", sort=True, observed=True)
    }
    rows: list[dict] = []
    for iteration in range(cfg.null_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        returns = []
        for row in portfolio.itertuples(index=False):
            local = groups[pd.Timestamp(row.entry_time)]
            usable = local.dropna(subset=["price_return", "future_funding", "btc_beta"])
            symbols = np.asarray(sorted(usable["symbol"].astype(str).unique()))
            long_count = int(row.long_side_breadth)
            short_count = int(row.short_side_breadth)
            chosen = symbols[
                rng.choice(len(symbols), size=long_count + short_count, replace=False)
            ]
            distribution = {
                str(symbol): 0.5 / long_count for symbol in chosen[:long_count]
            }
            distribution.update(
                {
                    str(symbol): -0.5 / short_count
                    for symbol in chosen[long_count:]
                }
            )
            _, components = beta_neutral_components(local, distribution)
            returns.append(
                components["gross_return"]
                - cfg.one_way_cost * float(row.realized_turnover)
            )
        rows.append(
            {
                "iteration": iteration,
                "candidate": CANDIDATE,
                "null_type": "random_full_universe_observed_side_breadth_and_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def summarize_v148(
    portfolio: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V148Config = V148Config(),
) -> pd.DataFrame:
    sample = portfolio.sort_values("entry_time")
    values = sample["primary_net_return"].to_numpy(dtype=float)
    draws = _moving_block_means(
        values,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + 2),
    )
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = sample.groupby("period", observed=True)["primary_net_return"].mean()
    states = sample.groupby("breadth_state", observed=True)["primary_net_return"].mean()
    months = sample.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    counts = sample["period"].value_counts()
    observed = float(values.mean())
    row = {
        "candidate": CANDIDATE,
        "weeks": len(sample),
        "months": sample["month_start"].nunique(),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "contracted_weeks": int(sample["breadth_state"].eq("contracted4to8").sum()),
        "median_long_breadth": sample["long_side_breadth"].median(),
        "median_short_breadth": sample["short_side_breadth"].median(),
        "mean_retained_opposite_names": sample["retained_opposite_sign_names"].mean(),
        "mean_turnover": sample["realized_turnover"].mean(),
        "mean_price_bp": sample["price_return"].mean() * 10_000,
        "mean_funding_bp": sample["funding_return"].mean() * 10_000,
        "mean_gross_bp": sample["gross_return"].mean() * 10_000,
        "mean_primary_net_bp": observed * 10_000,
        "mean_stress_net_bp": sample["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "contracted_primary_net_bp": states.get("contracted4to8", np.nan) * 10_000,
        "broad_primary_net_bp": states.get("broad9plus", np.nan) * 10_000,
        "bootstrap_95_low_bp": ci_low * 10_000,
        "bootstrap_95_high_bp": ci_high * 10_000,
        "null_percentile": 100 * nulls["mean_primary_net_return"].le(observed).mean(),
        "positive_month_concentration": concentration,
        "worst_period_bp": periods.min() * 10_000,
        "max_abs_residual_btc_beta": sample["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (sample["gross_notional"] - 1.0).abs().max(),
    }
    row["promote"] = bool(
        row["weeks"] >= 45
        and row["months"] >= 11
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 10
        and row["mean_turnover"] <= 0.75
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
        and row["null_percentile"] >= 95
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and row["max_abs_residual_btc_beta"] <= 1e-12
        and row["max_gross_notional_drift"] <= 1e-12
    )
    return pd.DataFrame([row])


def compare_v148_to_v147(
    portfolio: pd.DataFrame,
    cfg: V148Config = V148Config(),
) -> pd.DataFrame:
    base = pd.read_parquet(cfg.v147_portfolio_path)
    base = base.loc[
        base["candidate"].eq("FSS1_ALL_NEGATIVE_LONG_ALL_POSITIVE_SHORT"),
        ["entry_time", "primary_net_return", "realized_turnover"],
    ].rename(
        columns={
            "primary_net_return": "v147_return",
            "realized_turnover": "v147_turnover",
        }
    )
    merged = portfolio[
        ["entry_time", "primary_net_return", "realized_turnover"]
    ].merge(base, on="entry_time", how="inner")
    return pd.DataFrame(
        [
            {
                "overlap_weeks": len(merged),
                "return_correlation": merged[
                    ["primary_net_return", "v147_return"]
                ].corr().iloc[0, 1],
                "mean_return_difference_bp": (
                    merged["primary_net_return"] - merged["v147_return"]
                ).mean()
                * 10_000,
                "mean_turnover_reduction": (
                    merged["v147_turnover"] - merged["realized_turnover"]
                ).mean(),
            }
        ]
    )


def write_v148_funding_sign_hysteresis(
    cfg: V148Config = V148Config(),
) -> dict[str, Path]:
    panel = load_v147_panel(V147Config(panel_path=cfg.panel_path))
    portfolio = build_v148_portfolio(panel, cfg)
    nulls = build_v148_nulls(panel, portfolio, cfg)
    summary = summarize_v148(portfolio, nulls, cfg)
    comparison = compare_v148_to_v147(portfolio, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "comparison": root / "v147_comparison.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    portfolio.drop(columns="_weights").to_parquet(outputs["portfolio"], index=False)
    nulls.to_csv(outputs["nulls"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    comparison.to_csv(outputs["comparison"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    outputs["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "weeks": len(portfolio),
                "null_iterations": cfg.null_iterations,
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_candidate"
    outputs["findings"].write_text(
        "\n".join(
            [
                "# v14.8 Funding-Sign Hysteresis Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## v14.7 comparison",
                "",
                comparison.to_markdown(index=False, floatfmt=".4f"),
                "",
                "No PaperLive, leverage, or real-order permission changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return outputs
