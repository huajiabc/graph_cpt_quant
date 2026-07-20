"""Turnover-governed hold-band implementation of cross-venue carry."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


PANEL_PATH = Path(
    "reports/v12_5_cross_venue_perpetual_carry/weekly_symbol_panel.parquet"
)
REPORT_ROOT = Path("reports/v12_6_turnover_governed_cross_venue_carry")
CANDIDATE = "TG1_30D_TOP9_HOLD18"


@dataclass(frozen=True)
class V126Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    bucket_size: int = 9
    hold_rank: int = 18
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def load_v126_panel(cfg: V126Config = V126Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    return panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _select_hold_band(
    local: pd.DataFrame,
    previous: list[str],
    bucket_size: int,
    hold_rank: int,
) -> list[str]:
    ranked = local.dropna(subset=["score_30d", "pair_gross_return"])
    ranked = ranked[ranked["score_30d"].gt(0)].sort_values(
        ["score_30d", "symbol"], ascending=[False, True]
    )
    if len(ranked) < bucket_size:
        return []
    ranks = {
        str(symbol): rank
        for rank, symbol in enumerate(ranked["symbol"].astype(str), start=1)
    }
    retained = [
        symbol for symbol in previous if symbol in ranks and ranks[symbol] <= hold_rank
    ]
    selected = retained[:bucket_size]
    for symbol in ranked["symbol"].astype(str):
        if len(selected) >= bucket_size:
            break
        if symbol not in selected:
            selected.append(symbol)
    return selected


def _components(local: pd.DataFrame, selected: list[str]) -> dict[str, float]:
    indexed = local.set_index("symbol")
    return {
        name: float(indexed.loc[selected, column].mean())
        for name, column in (
            ("bybit_return", "bybit_return"),
            ("binance_return", "binance_return"),
            ("price_basis_return", "price_basis_return"),
            ("funding_spread_return", "funding_spread_return"),
            ("gross_return", "pair_gross_return"),
        )
    }


def build_v126_portfolio(
    panel: pd.DataFrame, cfg: V126Config = V126Config()
) -> pd.DataFrame:
    rows = []
    previous: list[str] = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        selected = _select_hold_band(
            local, previous, cfg.bucket_size, cfg.hold_rank
        )
        if not selected:
            previous = []
            previous_weights = None
            continue
        weights = {symbol: 1.0 / cfg.bucket_size for symbol in selected}
        if previous_weights is None:
            turnover = 1.0
        else:
            symbols = set(previous_weights) | set(weights)
            turnover = sum(
                abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in symbols
            )
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "selected_symbols": "|".join(selected),
                "retained_names": len(set(selected) & set(previous)),
                "realized_turnover": turnover,
                **_components(local, selected),
            }
        )
        previous = selected
        previous_weights = weights
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output.loc[output.index[-1], "realized_turnover"] += 1.0
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"]
        - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    return output


def build_v126_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V126Config = V126Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    weekly = {
        pd.Timestamp(entry): group[
            group["score_30d"].gt(0) & group["pair_gross_return"].notna()
        ]["pair_gross_return"].to_numpy(dtype=float)
        for entry, group in panel.groupby("entry_time", sort=True, observed=True)
    }
    costs = (
        cfg.one_way_cost * portfolio.set_index("entry_time")["realized_turnover"]
    )
    rows = []
    for iteration in range(cfg.null_iterations):
        returns = []
        for entry, cost in costs.items():
            values = weekly[pd.Timestamp(entry)]
            if len(values) < cfg.bucket_size:
                continue
            gross = rng.choice(values, size=cfg.bucket_size, replace=False).mean()
            returns.append(float(gross - cost))
        rows.append(
            {
                "iteration": iteration,
                "null_type": "random_positive_spread_fixed_observed_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def summarize_v126(
    portfolio: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V126Config = V126Config(),
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
        "median_coverage": float(portfolio["coverage"].median()),
        "mean_turnover": float(portfolio["realized_turnover"].mean()),
        "mean_retained_names": float(portfolio["retained_names"].mean()),
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
        "null_percentile": float(
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
        and row["mean_turnover"] <= 0.50
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
        and row["null_percentile"] >= 90
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
    )
    return pd.DataFrame([row])


def write_v126_turnover_governed_cross_venue_carry(
    cfg: V126Config = V126Config(),
) -> dict[str, Path]:
    panel = load_v126_panel(cfg)
    portfolio = build_v126_portfolio(panel, cfg)
    nulls = build_v126_nulls(panel, portfolio, cfg)
    summary = summarize_v126(portfolio, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v126_turnover_governed_cross_venue_carry_findings_2026_07_15.md"
        ),
    }
    portfolio.to_parquet(paths["portfolio"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "panel_rows": len(panel),
                "weeks": len(portfolio),
                "candidate": CANDIDATE,
                "promoted": [CANDIDATE] if promoted else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_candidate" if promoted else "reject_as_tradable_alpha"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v12.6 Turnover-Governed Cross-Venue Carry Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The score, venue orientation, hold band, turnover costs, and controls "
                "were frozen before inspecting this portfolio return. No existing "
                "PaperLive strategy was changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
