"""Turnover-capped execution of the v14.7 funding-sign spread."""
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
from pressure_graph.reports.v147_funding_sign_spread import (
    PANEL_PATH,
    REPORT_ROOT as V147_REPORT_ROOT,
    V147Config,
    beta_neutral_components,
    load_v147_panel,
)


REPORT_ROOT = Path("reports/v14_9_funding_sign_turnover_cap")
FINDINGS_PATH = Path("docs/v149_funding_sign_turnover_cap_findings_2026_07_15.md")
CANDIDATE = "FSS3_CURRENT_SIGN_070_TURNOVER_CAP"


@dataclass(frozen=True)
class V149Config:
    panel_path: Path = PANEL_PATH
    v147_portfolio_path: Path = V147_REPORT_ROOT / "weekly_portfolios.parquet"
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_side_breadth: int = 4
    transition_turnover_cap: float = 0.70
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715
    bisection_iterations: int = 48


def weight_turnover(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        sum(
            abs(left.get(symbol, 0.0) - right.get(symbol, 0.0))
            for symbol in set(left) | set(right)
        )
    )


def funding_sign_target(
    local: pd.DataFrame,
    minimum_side_breadth: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], int, int]:
    eligible = local.dropna(
        subset=["score_7d", "price_return", "future_funding", "btc_beta"]
    )
    negative = sorted(
        eligible.loc[eligible["score_7d"].lt(0), "symbol"].astype(str).unique()
    )
    positive = sorted(
        eligible.loc[eligible["score_7d"].gt(0), "symbol"].astype(str).unique()
    )
    if len(negative) < minimum_side_breadth or len(positive) < minimum_side_breadth:
        return {}, {}, {}, len(negative), len(positive)
    raw = {symbol: 0.5 / len(negative) for symbol in negative}
    raw.update({symbol: -0.5 / len(positive) for symbol in positive})
    target, components = beta_neutral_components(local, raw)
    return target, components, raw, len(negative), len(positive)


def _blend_and_neutralize(
    local: pd.DataFrame,
    previous_weights: dict[str, float],
    target_weights: dict[str, float],
    fraction: float,
) -> tuple[dict[str, float], dict[str, float]]:
    current_symbols = set(local["symbol"].astype(str))
    previous_alt = {
        symbol: weight
        for symbol, weight in previous_weights.items()
        if symbol != BTC and symbol in current_symbols
    }
    target_alt = {
        symbol: weight for symbol, weight in target_weights.items() if symbol != BTC
    }
    alt = {
        symbol: (1.0 - fraction) * previous_alt.get(symbol, 0.0)
        + fraction * target_alt.get(symbol, 0.0)
        for symbol in set(previous_alt) | set(target_alt)
    }
    alt = {symbol: weight for symbol, weight in alt.items() if abs(weight) > 1e-16}
    return beta_neutral_components(local, alt)


def execute_capped_transition(
    local: pd.DataFrame,
    previous_weights: dict[str, float] | None,
    target_weights: dict[str, float],
    cap: float,
    bisection_iterations: int = 48,
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    """Return weights, PnL components, target fraction, turnover and cap breach."""
    if previous_weights is None:
        _, components = _blend_and_neutralize(local, {}, target_weights, 1.0)
        return target_weights, components, 1.0, weight_turnover({}, target_weights), 0.0

    target_turnover = weight_turnover(previous_weights, target_weights)
    if target_turnover <= cap + 1e-14:
        _, components = _blend_and_neutralize(
            local, previous_weights, target_weights, 1.0
        )
        return target_weights, components, 1.0, target_turnover, 0.0

    base_weights, base_components = _blend_and_neutralize(
        local, previous_weights, target_weights, 0.0
    )
    base_turnover = weight_turnover(previous_weights, base_weights)
    if base_turnover >= cap:
        return (
            base_weights,
            base_components,
            0.0,
            base_turnover,
            max(0.0, base_turnover - cap),
        )

    low = 0.0
    high = 1.0
    best_weights = base_weights
    best_components = base_components
    best_turnover = base_turnover
    for _ in range(bisection_iterations):
        middle = 0.5 * (low + high)
        weights, components = _blend_and_neutralize(
            local, previous_weights, target_weights, middle
        )
        turnover = weight_turnover(previous_weights, weights)
        if turnover <= cap:
            low = middle
            best_weights = weights
            best_components = components
            best_turnover = turnover
        else:
            high = middle
    return best_weights, best_components, low, best_turnover, max(0.0, best_turnover - cap)


def build_v149_portfolio(
    panel: pd.DataFrame,
    cfg: V149Config = V149Config(),
) -> pd.DataFrame:
    rows: list[dict] = []
    previous_weights: dict[str, float] | None = None
    previous_entry: pd.Timestamp | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        entry = pd.Timestamp(entry)
        target, _, _, negative_breadth, positive_breadth = funding_sign_target(
            local, cfg.minimum_side_breadth
        )
        has_gap = previous_entry is not None and entry - previous_entry > pd.Timedelta(
            days=7, minutes=1
        )
        if has_gap and previous_weights is not None:
            rows[-1]["realized_turnover"] += sum(abs(w) for w in previous_weights.values())
            previous_weights = None
        if not target:
            if previous_weights is not None:
                rows[-1]["realized_turnover"] += sum(
                    abs(w) for w in previous_weights.values()
                )
                previous_weights = None
            previous_entry = entry
            continue
        weights, components, fraction, rebalance_turnover, cap_breach = (
            execute_capped_transition(
                local,
                previous_weights,
                target,
                cfg.transition_turnover_cap,
                cfg.bisection_iterations,
            )
        )
        target_tracking_l1 = weight_turnover(weights, target)
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "negative_breadth": negative_breadth,
                "positive_breadth": positive_breadth,
                "breadth_state": (
                    "contracted4to8" if negative_breadth <= 8 else "broad9plus"
                ),
                "selected_long_symbols": "|".join(
                    sorted(symbol for symbol, weight in weights.items() if symbol != BTC and weight > 0)
                ),
                "selected_short_symbols": "|".join(
                    sorted(symbol for symbol, weight in weights.items() if symbol != BTC and weight < 0)
                ),
                "executed_target_fraction": fraction,
                "target_tracking_l1": target_tracking_l1,
                "cap_applicable": previous_weights is not None,
                "cap_binding": previous_weights is not None and fraction < 1.0 - 1e-10,
                "rebalance_turnover": rebalance_turnover,
                "cap_breach": cap_breach,
                "realized_turnover": rebalance_turnover,
                "_weights": weights,
                **components,
            }
        )
        previous_weights = weights
        previous_entry = entry
    if rows and previous_weights is not None:
        rows[-1]["realized_turnover"] += sum(abs(w) for w in previous_weights.values())
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


def build_v149_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V149Config = V149Config(),
) -> pd.DataFrame:
    observed = portfolio.sort_values("entry_time").reset_index(drop=True)
    all_symbols = sorted(panel["symbol"].astype(str).unique())
    symbol_to_index = {symbol: index for index, symbol in enumerate(all_symbols)}
    groups = {
        pd.Timestamp(entry): local
        for entry, local in panel.groupby("entry_time", sort=True, observed=True)
    }
    week_arrays = []
    for row in observed.itertuples(index=False):
        local = groups[pd.Timestamp(row.entry_time)]
        usable = local.dropna(subset=["price_return", "future_funding", "btc_beta"])
        indexed = usable.set_index("symbol")
        symbols = sorted(usable["symbol"].astype(str).unique())
        indices = np.asarray([symbol_to_index[symbol] for symbol in symbols], dtype=int)
        beta = np.zeros(len(all_symbols), dtype=float)
        price = np.zeros(len(all_symbols), dtype=float)
        funding = np.zeros(len(all_symbols), dtype=float)
        beta[indices] = indexed.loc[symbols, "btc_beta"].to_numpy(dtype=float)
        price[indices] = indexed.loc[symbols, "price_return"].to_numpy(dtype=float)
        funding[indices] = indexed.loc[symbols, "future_funding"].to_numpy(dtype=float)
        mask = np.zeros(len(all_symbols), dtype=bool)
        mask[indices] = True
        week_arrays.append(
            {
                "indices": indices,
                "beta": beta,
                "price": price,
                "funding": funding,
                "mask": mask,
                "btc_return": float(usable.iloc[0]["btc_return"]),
                "btc_funding": float(usable.iloc[0]["btc_future_funding"]),
                "long_count": int(row.negative_breadth),
                "short_count": int(row.positive_breadth),
            }
        )
    rows: list[dict] = []
    for iteration in range(cfg.null_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        previous_alt: np.ndarray | None = None
        previous_btc = 0.0
        gross_returns: list[float] = []
        turnovers: list[float] = []
        for week in week_arrays:
            indices = week["indices"]
            long_count = week["long_count"]
            short_count = week["short_count"]
            chosen = indices[
                rng.choice(len(indices), size=long_count + short_count, replace=False)
            ]
            raw = np.zeros(len(all_symbols), dtype=float)
            raw[chosen[:long_count]] = 0.5 / long_count
            raw[chosen[long_count:]] = -0.5 / short_count
            target_alt, target_btc = _neutralize_alt_array(raw, week["beta"])
            executed_alt, executed_btc, turnover = _execute_capped_array_transition(
                previous_alt,
                previous_btc,
                target_alt,
                target_btc,
                week["beta"],
                week["mask"],
                cfg.transition_turnover_cap,
                cfg.bisection_iterations,
            )
            price_return = float(
                np.dot(executed_alt, week["price"])
                + executed_btc * week["btc_return"]
            )
            funding_return = float(
                -np.dot(executed_alt, week["funding"])
                - executed_btc * week["btc_funding"]
            )
            gross_returns.append(price_return + funding_return)
            turnovers.append(turnover)
            previous_alt = executed_alt
            previous_btc = executed_btc
        if turnovers and previous_alt is not None:
            turnovers[-1] += float(np.abs(previous_alt).sum() + abs(previous_btc))
        primary = np.asarray(gross_returns) - cfg.one_way_cost * np.asarray(turnovers)
        rows.append(
            {
                "iteration": iteration,
                "candidate": CANDIDATE,
                "null_type": "random_sign_breadth_identical_capped_execution",
                "mean_primary_net_return": float(primary.mean()),
            }
        )
    return pd.DataFrame(rows)


def _neutralize_alt_array(
    alt_weights: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, float]:
    hedge = -float(np.dot(alt_weights, beta))
    gross = float(np.abs(alt_weights).sum() + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return np.zeros_like(alt_weights), 0.0
    return alt_weights / gross, hedge / gross


def _execute_capped_array_transition(
    previous_alt: np.ndarray | None,
    previous_btc: float,
    target_alt: np.ndarray,
    target_btc: float,
    beta: np.ndarray,
    current_mask: np.ndarray,
    cap: float,
    bisection_iterations: int,
) -> tuple[np.ndarray, float, float]:
    if previous_alt is None:
        return target_alt, target_btc, float(np.abs(target_alt).sum() + abs(target_btc))

    def turnover(alt: np.ndarray, btc: float) -> float:
        return float(np.abs(alt - previous_alt).sum() + abs(btc - previous_btc))

    target_turnover = turnover(target_alt, target_btc)
    if target_turnover <= cap + 1e-14:
        return target_alt, target_btc, target_turnover

    previous_current = previous_alt * current_mask

    def point(fraction: float) -> tuple[np.ndarray, float, float]:
        raw = (1.0 - fraction) * previous_current + fraction * target_alt
        alt, btc = _neutralize_alt_array(raw, beta)
        return alt, btc, turnover(alt, btc)

    base_alt, base_btc, base_turnover = point(0.0)
    if base_turnover >= cap:
        return base_alt, base_btc, base_turnover

    low = 0.0
    high = 1.0
    best_alt = base_alt
    best_btc = base_btc
    best_turnover = base_turnover
    for _ in range(bisection_iterations):
        middle = 0.5 * (low + high)
        alt, btc, current_turnover = point(middle)
        if current_turnover <= cap:
            low = middle
            best_alt = alt
            best_btc = btc
            best_turnover = current_turnover
        else:
            high = middle
    return best_alt, best_btc, best_turnover


def summarize_v149(
    portfolio: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V149Config = V149Config(),
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
    applicable = sample[sample["cap_applicable"]]
    observed_mean = float(values.mean())
    row = {
        "candidate": CANDIDATE,
        "weeks": len(sample),
        "months": sample["month_start"].nunique(),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "contracted_weeks": int(sample["breadth_state"].eq("contracted4to8").sum()),
        "median_negative_breadth": sample["negative_breadth"].median(),
        "median_positive_breadth": sample["positive_breadth"].median(),
        "mean_turnover": sample["realized_turnover"].mean(),
        "cap_binding_weeks": int(sample["cap_binding"].sum()),
        "mean_executed_target_fraction": sample["executed_target_fraction"].mean(),
        "mean_target_tracking_l1": sample["target_tracking_l1"].mean(),
        "max_capped_transition_turnover": applicable["rebalance_turnover"].max(),
        "max_cap_breach": sample["cap_breach"].max(),
        "mean_price_bp": sample["price_return"].mean() * 10_000,
        "mean_funding_bp": sample["funding_return"].mean() * 10_000,
        "mean_gross_bp": sample["gross_return"].mean() * 10_000,
        "mean_primary_net_bp": observed_mean * 10_000,
        "mean_stress_net_bp": sample["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "contracted_primary_net_bp": states.get("contracted4to8", np.nan) * 10_000,
        "broad_primary_net_bp": states.get("broad9plus", np.nan) * 10_000,
        "bootstrap_95_low_bp": ci_low * 10_000,
        "bootstrap_95_high_bp": ci_high * 10_000,
        "null_percentile": 100 * nulls["mean_primary_net_return"].le(observed_mean).mean(),
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
        and row["max_capped_transition_turnover"] <= cfg.transition_turnover_cap + 1e-10
        and row["max_cap_breach"] <= 1e-10
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


def compare_v149_to_v147(
    portfolio: pd.DataFrame,
    cfg: V149Config = V149Config(),
) -> pd.DataFrame:
    base = pd.read_parquet(cfg.v147_portfolio_path)
    base = base.loc[
        base["candidate"].eq("FSS1_ALL_NEGATIVE_LONG_ALL_POSITIVE_SHORT"),
        ["entry_time", "primary_net_return", "realized_turnover"],
    ].rename(columns={"primary_net_return": "v147_return", "realized_turnover": "v147_turnover"})
    merged = portfolio[["entry_time", "primary_net_return", "realized_turnover"]].merge(
        base, on="entry_time", how="inner"
    )
    return pd.DataFrame(
        [
            {
                "overlap_weeks": len(merged),
                "return_correlation": merged[["primary_net_return", "v147_return"]].corr().iloc[0, 1],
                "mean_return_difference_bp": (merged["primary_net_return"] - merged["v147_return"]).mean() * 10_000,
                "mean_turnover_reduction": (merged["v147_turnover"] - merged["realized_turnover"]).mean(),
            }
        ]
    )


def write_v149_funding_sign_turnover_cap(
    cfg: V149Config = V149Config(),
) -> dict[str, Path]:
    panel = load_v147_panel(V147Config(panel_path=cfg.panel_path))
    portfolio = build_v149_portfolio(panel, cfg)
    nulls = build_v149_nulls(panel, portfolio, cfg)
    summary = summarize_v149(portfolio, nulls, cfg)
    comparison = compare_v149_to_v147(portfolio, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "portfolio": root / "weekly_portfolio.parquet",
        "weights": root / "weekly_weights.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "comparison": root / "v147_comparison.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    weight_rows = [
        {
            "entry_time": row["entry_time"],
            "symbol": symbol,
            "weight": weight,
            "is_btc_hedge": symbol == BTC,
        }
        for _, row in portfolio.iterrows()
        for symbol, weight in row["_weights"].items()
    ]
    pd.DataFrame(weight_rows).to_parquet(outputs["weights"], index=False)
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
                "turnover_cap": cfg.transition_turnover_cap,
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
                "# v14.9 Funding-Sign Turnover-Cap Findings",
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
