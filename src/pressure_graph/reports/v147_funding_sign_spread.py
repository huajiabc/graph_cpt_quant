"""Equal-weight funding-sign portfolios with exact BTC-beta neutrality."""
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
from pressure_graph.reports.v135_adaptive_negative_funding_breadth import (
    PANEL_PATH,
)
from pressure_graph.reports.v140_equal_weight_negative_funding_state import (
    REPORT_ROOT as V140_REPORT_ROOT,
)


REPORT_ROOT = Path("reports/v14_7_funding_sign_spread")
FINDINGS_PATH = Path("docs/v147_funding_sign_spread_findings_2026_07_15.md")
CANDIDATES = (
    "FSS1_ALL_NEGATIVE_LONG_ALL_POSITIVE_SHORT",
    "PFS1_ALL_POSITIVE_SHORT",
)


@dataclass(frozen=True)
class V147Config:
    panel_path: Path = PANEL_PATH
    nf8_portfolio_path: Path = V140_REPORT_ROOT / "weekly_portfolio.parquet"
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_negative_breadth: int = 4
    minimum_positive_breadth: int = 4
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def load_v147_panel(cfg: V147Config = V147Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    return panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def sign_distributions(
    local: pd.DataFrame,
    cfg: V147Config,
) -> tuple[dict[str, float], dict[str, float], int, int]:
    eligible = local.dropna(
        subset=["score_7d", "price_return", "future_funding", "btc_beta"]
    )
    negative = sorted(
        eligible.loc[eligible["score_7d"].lt(0), "symbol"].astype(str).unique()
    )
    positive = sorted(
        eligible.loc[eligible["score_7d"].gt(0), "symbol"].astype(str).unique()
    )
    if (
        len(negative) < cfg.minimum_negative_breadth
        or len(positive) < cfg.minimum_positive_breadth
    ):
        return {}, {}, len(negative), len(positive)
    spread = {symbol: 0.5 / len(negative) for symbol in negative}
    spread.update({symbol: -0.5 / len(positive) for symbol in positive})
    positive_short = {symbol: -1.0 / len(positive) for symbol in positive}
    return spread, positive_short, len(negative), len(positive)


def beta_neutral_components(
    local: pd.DataFrame,
    raw_alt_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    indexed = local.set_index("symbol")
    if not raw_alt_weights:
        return {}, {}
    alt_beta = float(
        sum(
            raw_alt_weights[symbol] * float(indexed.at[symbol, "btc_beta"])
            for symbol in raw_alt_weights
        )
    )
    unscaled = dict(raw_alt_weights)
    unscaled[BTC] = -alt_beta
    gross = float(sum(abs(weight) for weight in unscaled.values()))
    if not np.isfinite(gross) or gross <= 0:
        return {}, {}
    weights = {symbol: weight / gross for symbol, weight in unscaled.items()}
    alt_price = float(
        sum(
            weights[symbol] * float(indexed.at[symbol, "price_return"])
            for symbol in raw_alt_weights
        )
    )
    btc_price = float(weights[BTC] * float(indexed.iloc[0]["btc_return"]))
    alt_funding = float(
        sum(
            -weights[symbol] * float(indexed.at[symbol, "future_funding"])
            for symbol in raw_alt_weights
        )
    )
    btc_funding = float(
        -weights[BTC] * float(indexed.iloc[0]["btc_future_funding"])
    )
    residual_beta = float(
        sum(
            weights[symbol] * float(indexed.at[symbol, "btc_beta"])
            for symbol in raw_alt_weights
        )
        + weights[BTC]
    )
    return weights, {
        "alt_long_notional": float(
            sum(max(weights[symbol], 0.0) for symbol in raw_alt_weights)
        ),
        "alt_short_notional": float(
            sum(-min(weights[symbol], 0.0) for symbol in raw_alt_weights)
        ),
        "btc_hedge_weight": float(weights[BTC]),
        "price_return": alt_price + btc_price,
        "alt_funding_return": alt_funding,
        "btc_funding_return": btc_funding,
        "funding_return": alt_funding + btc_funding,
        "gross_return": alt_price + btc_price + alt_funding + btc_funding,
        "gross_notional": float(sum(abs(weight) for weight in weights.values())),
        "residual_btc_beta": residual_beta,
    }


def _turnover_with_terminal_close(
    rows: list[dict],
    candidate: str,
) -> None:
    indices = [index for index, row in enumerate(rows) if row["candidate"] == candidate]
    previous: dict[str, float] | None = None
    previous_entry: pd.Timestamp | None = None
    for index in indices:
        current = rows[index]["_weights"]
        entry = pd.Timestamp(rows[index]["entry_time"])
        if previous is None or (
            previous_entry is not None
            and entry - previous_entry > pd.Timedelta(days=7, minutes=1)
        ):
            if previous is not None:
                rows[indices[indices.index(index) - 1]]["realized_turnover"] += sum(
                    abs(weight) for weight in previous.values()
                )
            turnover = sum(abs(weight) for weight in current.values())
        else:
            turnover = sum(
                abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
                for symbol in set(current) | set(previous)
            )
        rows[index]["realized_turnover"] = float(turnover)
        previous = current
        previous_entry = entry
    if indices and previous is not None:
        rows[indices[-1]]["realized_turnover"] += sum(
            abs(weight) for weight in previous.values()
        )


def build_v147_portfolios(
    panel: pd.DataFrame,
    cfg: V147Config = V147Config(),
) -> pd.DataFrame:
    rows: list[dict] = []
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        spread, positive_short, negative_breadth, positive_breadth = (
            sign_distributions(local, cfg)
        )
        distributions = {
            CANDIDATES[0]: spread,
            CANDIDATES[1]: positive_short,
        }
        for candidate, distribution in distributions.items():
            weights, components = beta_neutral_components(local, distribution)
            if not weights:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "entry_time": entry,
                    "exit_time": local["exit_time"].iloc[0],
                    "month_start": local["month_start"].iloc[0],
                    "period": local["period"].iloc[0],
                    "coverage": len(local),
                    "negative_breadth": negative_breadth,
                    "positive_breadth": positive_breadth,
                    "breadth_state": "contracted4to8"
                    if negative_breadth <= 8
                    else "broad9plus",
                    "selected_long_symbols": "|".join(
                        sorted(symbol for symbol, weight in distribution.items() if weight > 0)
                    ),
                    "selected_short_symbols": "|".join(
                        sorted(symbol for symbol, weight in distribution.items() if weight < 0)
                    ),
                    "realized_turnover": 0.0,
                    "_weights": weights,
                    **components,
                }
            )
    for candidate in CANDIDATES:
        _turnover_with_terminal_close(rows, candidate)
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


def build_v147_nulls(
    panel: pd.DataFrame,
    portfolios: pd.DataFrame,
    cfg: V147Config = V147Config(),
) -> pd.DataFrame:
    groups = {
        pd.Timestamp(entry): local
        for entry, local in panel.groupby("entry_time", sort=True, observed=True)
    }
    observed = {
        candidate: group.sort_values("entry_time").reset_index(drop=True)
        for candidate, group in portfolios.groupby("candidate", sort=True)
    }
    rows = []
    for iteration in range(cfg.null_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        means = {}
        for candidate in CANDIDATES:
            returns = []
            for row in observed[candidate].itertuples(index=False):
                local = groups[pd.Timestamp(row.entry_time)]
                usable = local.dropna(
                    subset=["price_return", "future_funding", "btc_beta"]
                )
                symbols = np.asarray(sorted(usable["symbol"].astype(str).unique()))
                negative_breadth = int(row.negative_breadth)
                positive_breadth = int(row.positive_breadth)
                if candidate == CANDIDATES[0]:
                    take = negative_breadth + positive_breadth
                    chosen = symbols[rng.choice(len(symbols), size=take, replace=False)]
                    long_symbols = chosen[:negative_breadth]
                    short_symbols = chosen[negative_breadth:]
                    distribution = {
                        str(symbol): 0.5 / negative_breadth for symbol in long_symbols
                    }
                    distribution.update(
                        {
                            str(symbol): -0.5 / positive_breadth
                            for symbol in short_symbols
                        }
                    )
                else:
                    chosen = symbols[
                        rng.choice(len(symbols), size=positive_breadth, replace=False)
                    ]
                    distribution = {
                        str(symbol): -1.0 / positive_breadth for symbol in chosen
                    }
                _, components = beta_neutral_components(local, distribution)
                returns.append(
                    components["gross_return"]
                    - cfg.one_way_cost * float(row.realized_turnover)
                )
            mean = float(np.mean(returns))
            means[candidate] = mean
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "null_type": "random_full_universe_sign_breadth_observed_cost",
                    "mean_primary_net_return": mean,
                }
            )
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "null_type": "random_family_max",
                "mean_primary_net_return": max(means.values()),
            }
        )
    return pd.DataFrame(rows)


def summarize_v147(
    portfolios: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V147Config = V147Config(),
) -> pd.DataFrame:
    family = nulls.loc[
        nulls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
    ]
    rows = []
    for candidate in CANDIDATES:
        sample = portfolios[portfolios["candidate"].eq(candidate)].sort_values(
            "entry_time"
        )
        values = sample["primary_net_return"].to_numpy(dtype=float)
        draws = _moving_block_means(
            values,
            cfg.bootstrap_iterations,
            cfg.bootstrap_block_weeks,
            np.random.default_rng(cfg.seed + 2),
        )
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        periods = sample.groupby("period", observed=True)[
            "primary_net_return"
        ].mean()
        states = sample.groupby("breadth_state", observed=True)[
            "primary_net_return"
        ].mean()
        months = sample.groupby("month_start", observed=True)[
            "primary_net_return"
        ].sum()
        positive = months[months.gt(0)]
        concentration = (
            float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
        )
        observed_mean = float(values.mean())
        counts = sample["period"].value_counts()
        row = {
            "candidate": candidate,
            "weeks": len(sample),
            "months": sample["month_start"].nunique(),
            "validation_weeks": int(counts.get("validation", 0)),
            "holdout_weeks": int(counts.get("holdout", 0)),
            "contracted_weeks": int(sample["breadth_state"].eq("contracted4to8").sum()),
            "median_negative_breadth": sample["negative_breadth"].median(),
            "median_positive_breadth": sample["positive_breadth"].median(),
            "mean_turnover": sample["realized_turnover"].mean(),
            "mean_price_bp": sample["price_return"].mean() * 10_000,
            "mean_funding_bp": sample["funding_return"].mean() * 10_000,
            "mean_gross_bp": sample["gross_return"].mean() * 10_000,
            "mean_primary_net_bp": observed_mean * 10_000,
            "mean_stress_net_bp": sample["stress_net_return"].mean() * 10_000,
            "development_primary_net_bp": periods.get("development", np.nan)
            * 10_000,
            "validation_primary_net_bp": periods.get("validation", np.nan)
            * 10_000,
            "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
            "contracted_primary_net_bp": states.get("contracted4to8", np.nan)
            * 10_000,
            "broad_primary_net_bp": states.get("broad9plus", np.nan) * 10_000,
            "bootstrap_95_low_bp": ci_low * 10_000,
            "bootstrap_95_high_bp": ci_high * 10_000,
            "family_null_percentile": 100 * family.le(observed_mean).mean(),
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
            and row["family_null_percentile"] >= 95
            and row["positive_month_concentration"] <= 0.35
            and row["worst_period_bp"] >= -40
            and row["max_abs_residual_btc_beta"] <= 1e-12
            and row["max_gross_notional_drift"] <= 1e-12
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_v147_to_nf8(
    portfolios: pd.DataFrame,
    cfg: V147Config = V147Config(),
) -> pd.DataFrame:
    nf8 = pd.read_parquet(cfg.nf8_portfolio_path)[
        ["entry_time", "primary_net_return"]
    ].rename(columns={"primary_net_return": "nf8_return"})
    nf8["entry_time"] = pd.to_datetime(nf8["entry_time"], utc=True, errors="coerce")
    rows = []
    for candidate in CANDIDATES:
        sample = portfolios[portfolios["candidate"].eq(candidate)][
            ["entry_time", "primary_net_return"]
        ]
        merged = sample.merge(nf8, on="entry_time", how="inner")
        rows.append(
            {
                "candidate": candidate,
                "overlap_weeks": len(merged),
                "correlation_with_nf8": merged[
                    ["primary_net_return", "nf8_return"]
                ].corr().iloc[0, 1],
                "equal_weight_combo_mean_bp": 0.5
                * (merged["primary_net_return"] + merged["nf8_return"]).mean()
                * 10_000,
            }
        )
    return pd.DataFrame(rows)


def _write_findings(
    path: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_family"
    lines = [
        "# v14.7 Funding-Sign Spread Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## NF8 relationship",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "No PaperLive, leverage, or real-order permission changed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v147_funding_sign_spread(
    cfg: V147Config = V147Config(),
) -> dict[str, Path]:
    panel = load_v147_panel(cfg)
    portfolios = build_v147_portfolios(panel, cfg)
    nulls = build_v147_nulls(panel, portfolios, cfg)
    summary = summarize_v147(portfolios, nulls, cfg)
    comparison = compare_v147_to_nf8(portfolios, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "portfolios": root / "weekly_portfolios.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "comparison": root / "nf8_comparison.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    portfolios.drop(columns="_weights").to_parquet(outputs["portfolios"], index=False)
    nulls.to_csv(outputs["nulls"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    comparison.to_csv(outputs["comparison"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    outputs["metadata"].write_text(
        json.dumps(
            {
                "candidate_family": list(CANDIDATES),
                "weeks": int(portfolios["entry_time"].nunique()),
                "null_iterations": cfg.null_iterations,
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(cfg.findings_path, summary, comparison)
    return outputs
