"""Graph-diversified cross-venue carry with one held name per community."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v121_top_trader_community_rotation import _membership


PANEL_PATH = Path(
    "reports/v12_5_cross_venue_perpetual_carry/weekly_symbol_panel.parquet"
)
MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
REPORT_ROOT = Path("reports/v13_0_graph_diversified_cross_venue_carry")
CANDIDATE = "GC1_30D_COMMUNITY_TOP1_HOLD2"


@dataclass(frozen=True)
class V130Config:
    panel_path: Path = PANEL_PATH
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    community_count: int = 8
    minimum_communities: int = 6
    hold_rank: int = 2
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    random_partition_iterations: int = 200
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def _portfolio_path(
    panel: pd.DataFrame,
    group_column: str,
    cfg: V130Config,
) -> pd.DataFrame:
    rows = []
    previous: set[str] = set()
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        selected = []
        for _, group in local.groupby(group_column, observed=True):
            ranked = group.dropna(subset=["score_30d", "pair_gross_return"])
            ranked = ranked[ranked["score_30d"].gt(0)].sort_values(
                ["score_30d", "symbol"], ascending=[False, True]
            )
            if ranked.empty:
                continue
            ranked_symbols = ranked["symbol"].astype(str).tolist()
            retained = [
                symbol
                for symbol in ranked_symbols[: cfg.hold_rank]
                if symbol in previous
            ]
            selected.append(retained[0] if retained else ranked_symbols[0])
        if len(selected) < cfg.minimum_communities:
            previous = set()
            previous_weights = None
            continue
        weights = {
            symbol: 1.0 / cfg.community_count for symbol in sorted(selected)
        }
        if previous_weights is None:
            turnover = sum(weights.values())
        else:
            symbols = set(previous_weights) | set(weights)
            turnover = sum(
                abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in symbols
            )
        indexed = local.set_index("symbol")
        components = {
            name: float(
                sum(weights[symbol] * indexed.loc[symbol, column] for symbol in weights)
            )
            for name, column in (
                ("price_basis_return", "price_basis_return"),
                ("funding_spread_return", "funding_spread_return"),
                ("gross_return", "pair_gross_return"),
            )
        }
        rows.append(
            {
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "active_communities": len(weights),
                "invested_exposure": sum(weights.values()),
                "selected_symbols": "|".join(sorted(weights)),
                "realized_turnover": turnover,
                **components,
            }
        )
        previous = set(weights)
        previous_weights = weights
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output.loc[output.index[-1], "realized_turnover"] += output.loc[
        output.index[-1], "invested_exposure"
    ]
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"]
        - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    return output


def build_v130_portfolio(
    panel: pd.DataFrame, cfg: V130Config = V130Config()
) -> pd.DataFrame:
    output = _portfolio_path(panel, "community_id", cfg)
    output.insert(0, "candidate", CANDIDATE)
    return output


def build_v130_random_partitions(
    panel: pd.DataFrame, cfg: V130Config = V130Config()
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    membership = _membership(cfg)
    months = {
        pd.Timestamp(month): sorted(group["symbol"].astype(str).unique())
        for month, group in membership.groupby("month_start", observed=True)
    }
    rows = []
    for iteration in range(cfg.random_partition_iterations):
        assignments = {}
        for month, symbols in months.items():
            shuffled = np.asarray(symbols)[rng.permutation(len(symbols))]
            assignments[month] = {
                str(symbol): int(index // 9) for index, symbol in enumerate(shuffled)
            }
        randomized = panel.copy()
        randomized["random_community"] = [
            assignments[pd.Timestamp(month)][str(symbol)]
            for month, symbol in zip(
                randomized["month_start"], randomized["symbol"], strict=True
            )
        ]
        path = _portfolio_path(randomized, "random_community", cfg)
        rows.append(
            {
                "iteration": iteration,
                "weeks": len(path),
                "mean_primary_net_return": float(path["primary_net_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_v130(
    portfolio: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V130Config = V130Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 2)
    values = portfolio["primary_net_return"].to_numpy(dtype=float)
    draws = rng.choice(
        values, size=(cfg.bootstrap_iterations, len(values)), replace=True
    ).mean(axis=1)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    months = portfolio.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
    )
    observed = float(portfolio["primary_net_return"].mean())
    counts = portfolio["period"].value_counts()
    row = {
        "candidate": CANDIDATE,
        "weeks": len(portfolio),
        "months": int(portfolio["month_start"].nunique()),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "mean_invested_exposure": float(portfolio["invested_exposure"].mean()),
        "mean_turnover": float(portfolio["realized_turnover"].mean()),
        "mean_price_basis_bp": float(
            portfolio["price_basis_return"].mean() * 10_000
        ),
        "mean_funding_spread_bp": float(
            portfolio["funding_spread_return"].mean() * 10_000
        ),
        "mean_gross_bp": float(portfolio["gross_return"].mean() * 10_000),
        "mean_primary_net_bp": observed * 10_000,
        "mean_stress_net_bp": float(
            portfolio["stress_net_return"].mean() * 10_000
        ),
        "development_primary_net_bp": float(
            periods.get("development", np.nan) * 10_000
        ),
        "validation_primary_net_bp": float(
            periods.get("validation", np.nan) * 10_000
        ),
        "holdout_primary_net_bp": float(
            periods.get("holdout", np.nan) * 10_000
        ),
        "bootstrap_95_low_bp": float(ci_low * 10_000),
        "bootstrap_95_high_bp": float(ci_high * 10_000),
        "random_partition_percentile": float(
            100 * nulls["mean_primary_net_return"].le(observed).mean()
        ),
        "positive_month_concentration": concentration,
        "worst_period_bp": float(periods.min() * 10_000),
    }
    row["promote"] = bool(
        row["weeks"] >= 40
        and row["months"] >= 10
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 8
        and row["mean_invested_exposure"] >= 0.75
        and row["mean_funding_spread_bp"] > 0
        and all(
            row[key] > 0
            for key in (
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "mean_stress_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["random_partition_percentile"] >= 90
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
    )
    return pd.DataFrame([row])


def write_v130_graph_diversified_cross_venue_carry(
    cfg: V130Config = V130Config(),
) -> dict[str, Path]:
    panel = pd.read_parquet(cfg.panel_path)
    portfolio = build_v130_portfolio(panel, cfg)
    nulls = build_v130_random_partitions(panel, cfg)
    summary = summarize_v130(portfolio, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "random_partition_controls.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v130_graph_diversified_cross_venue_carry_findings_2026_07_15.md"
        ),
    }
    portfolio.to_parquet(paths["portfolio"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "weeks": len(portfolio),
                "promoted": [CANDIDATE] if promoted else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_graph_carry_shadow" if promoted else "reject_candidate"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v13.0 Graph-Diversified Cross-Venue Carry Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The frozen communities and random-partition controls use the same "
                "top-1/hold-2 implementation. PaperLive was not changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
