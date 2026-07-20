"""Beta-neutral negative-funding capitulation basket."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    BTC,
    estimate_v106_betas,
)
from pressure_graph.reports.v121_top_trader_community_rotation import _period
from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    RAW_BYBIT_ROOT,
    RECENT_ROOT,
    _combined_funding,
    hourly_bybit_prices,
    load_v132_bybit_klines,
)
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    MEMBERSHIP_PATH,
    _moving_block_means,
    load_v133_membership,
)


REPORT_ROOT = Path("reports/v13_4_negative_funding_beta_neutral_rebound")
CANDIDATE = "NF1_LOW9_HOLD18_BTC_BETA_NEUTRAL"


@dataclass(frozen=True)
class V134Config:
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    first_entry: pd.Timestamp = pd.Timestamp("2025-08-04", tz="UTC")
    bucket_size: int = 9
    hold_rank: int = 18
    holding_days: int = 7
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def _load_btc_funding() -> pd.DataFrame:
    frames = []
    for path in (
        RAW_BYBIT_ROOT / "funding" / f"{BTC}.parquet",
        RECENT_ROOT / "bybit_funding" / f"{BTC}.parquet",
    ):
        if path.exists():
            frame = pd.read_parquet(path)
            frame["symbol"] = BTC
            frames.append(frame[["symbol", "funding_time", "funding_rate_settled"]])
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    output["funding_time"] = pd.to_datetime(
        output["funding_time"], utc=True, errors="coerce"
    ).dt.floor("s")
    output["funding_rate_settled"] = pd.to_numeric(output["funding_rate_settled"], errors="coerce")
    return (
        output.dropna(subset=["funding_time", "funding_rate_settled"])
        .drop_duplicates(["symbol", "funding_time"], keep="last")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


def _monthly_betas(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    close = prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    rows = []
    for raw_month in sorted(membership["month_start"].unique()):
        month = pd.Timestamp(raw_month)
        symbols = set(membership.loc[membership["month_start"].eq(month), "symbol"].astype(str))
        history = returns.loc[
            (returns.index >= month - pd.Timedelta(days=30)) & (returns.index < month),
            sorted((symbols | {BTC}) & set(returns.columns.astype(str))),
        ]
        betas = estimate_v106_betas(history)
        for symbol in sorted(symbols & set(betas.index.astype(str))):
            rows.append({"month_start": month, "symbol": symbol, "btc_beta": betas[symbol]})
    return pd.DataFrame(rows)


def build_v134_weekly_panel(
    funding: pd.DataFrame,
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: V134Config = V134Config(),
) -> pd.DataFrame:
    close = prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    betas = _monthly_betas(prices, membership)
    beta_lookup = betas.set_index(["month_start", "symbol"])["btc_beta"]
    funding_groups = {
        str(symbol): frame.sort_values("funding_time")
        for symbol, frame in funding.groupby("symbol", observed=True)
    }
    month_symbols = {
        pd.Timestamp(month): sorted(frame["symbol"].astype(str).unique())
        for month, frame in membership.groupby("month_start", observed=True)
    }
    last_membership_end = membership["month_start"].max() + pd.offsets.MonthBegin(1)
    last_exit = min(close.index.max(), last_membership_end)
    entries = pd.date_range(
        cfg.first_entry,
        last_exit - pd.Timedelta(days=cfg.holding_days),
        freq="7D",
        tz="UTC",
    )
    btc_funding = funding_groups.get(BTC)
    if btc_funding is None or BTC not in close.columns:
        return pd.DataFrame()

    rows = []
    for entry in entries:
        exit_time = entry + pd.Timedelta(days=cfg.holding_days)
        if entry not in close.index or exit_time not in close.index:
            continue
        month = entry.replace(day=1)
        btc_rate = btc_funding["funding_rate_settled"]
        btc_time = btc_funding["funding_time"]
        btc_future_funding = float(btc_rate[btc_time.gt(entry) & btc_time.le(exit_time)].sum())
        btc_return = float(close.at[exit_time, BTC] / close.at[entry, BTC] - 1.0)
        for symbol in month_symbols.get(month, []):
            if symbol not in close.columns:
                continue
            entry_price = close.at[entry, symbol]
            exit_price = close.at[exit_time, symbol]
            if not all(np.isfinite(value) and value > 0 for value in (entry_price, exit_price)):
                continue
            symbol_funding = funding_groups.get(symbol)
            beta = beta_lookup.get((month, symbol), np.nan)
            if symbol_funding is None or not np.isfinite(beta):
                continue
            rate = symbol_funding["funding_rate_settled"]
            time = symbol_funding["funding_time"]
            score = rate[time.ge(entry - pd.Timedelta(days=7)) & time.lt(entry)]
            if score.empty:
                continue
            future_funding = float(rate[time.gt(entry) & time.le(exit_time)].sum())
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "month_start": month,
                    "period": _period(entry),
                    "symbol": symbol,
                    "score_7d": float(score.sum()),
                    "btc_beta": float(beta),
                    "price_return": float(exit_price / entry_price - 1.0),
                    "future_funding": future_funding,
                    "btc_return": btc_return,
                    "btc_future_funding": btc_future_funding,
                }
            )
    return pd.DataFrame(rows)


def _select_negative_hold_band(
    local: pd.DataFrame,
    previous: list[str],
    cfg: V134Config,
) -> list[str]:
    ranked = local.dropna(subset=["score_7d", "price_return", "btc_beta"])
    ranked = ranked[ranked["score_7d"].lt(0)].sort_values(
        ["score_7d", "symbol"], ascending=[True, True]
    )
    if len(ranked) < cfg.bucket_size:
        return []
    ranks = {str(symbol): rank for rank, symbol in enumerate(ranked["symbol"].astype(str), start=1)}
    selected = [
        symbol for symbol in previous if symbol in ranks and ranks[symbol] <= cfg.hold_rank
    ][: cfg.bucket_size]
    for symbol in ranked["symbol"].astype(str):
        if len(selected) >= cfg.bucket_size:
            break
        if symbol not in selected:
            selected.append(symbol)
    return selected


def _weights_and_components(
    local: pd.DataFrame,
    selected: list[str],
    cfg: V134Config,
) -> tuple[dict[str, float], dict[str, float]]:
    indexed = local.set_index("symbol")
    mean_beta = float(indexed.loc[selected, "btc_beta"].mean())
    if not np.isfinite(mean_beta) or mean_beta <= 0:
        return {}, {}
    long_total = 1.0 / (1.0 + mean_beta)
    btc_short = mean_beta / (1.0 + mean_beta)
    weights = {symbol: long_total / len(selected) for symbol in selected}
    weights[BTC] = -btc_short
    long_price = float(
        sum(weights[symbol] * indexed.at[symbol, "price_return"] for symbol in selected)
    )
    btc_price = float(weights[BTC] * indexed.iloc[0]["btc_return"])
    coin_funding = float(
        sum(-weights[symbol] * indexed.at[symbol, "future_funding"] for symbol in selected)
    )
    btc_funding = float(-weights[BTC] * indexed.iloc[0]["btc_future_funding"])
    residual_beta = float(
        sum(weights[symbol] * indexed.at[symbol, "btc_beta"] for symbol in selected) + weights[BTC]
    )
    return weights, {
        "selected_mean_beta": mean_beta,
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


def build_v134_portfolio(
    panel: pd.DataFrame,
    cfg: V134Config = V134Config(),
) -> pd.DataFrame:
    rows = []
    previous: list[str] = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        selected = _select_negative_hold_band(local, previous, cfg)
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
            if previous_weights is not None and rows:
                rows[-1]["realized_turnover"] += sum(
                    abs(weight) for weight in previous_weights.values()
                )
            previous = []
            previous_weights = None
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


def build_v134_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V134Config = V134Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    panel_groups = {
        pd.Timestamp(entry): frame
        for entry, frame in panel.groupby("entry_time", sort=True, observed=True)
    }
    observed_costs = portfolio.set_index("entry_time")["realized_turnover"]
    rows = []
    for iteration in range(cfg.null_iterations):
        returns = []
        for entry, turnover in observed_costs.items():
            local = panel_groups[pd.Timestamp(entry)]
            eligible = sorted(local.loc[local["score_7d"].lt(0), "symbol"].astype(str).unique())
            if len(eligible) < cfg.bucket_size:
                continue
            selected = list(
                np.asarray(eligible)[rng.choice(len(eligible), size=cfg.bucket_size, replace=False)]
            )
            _, components = _weights_and_components(local, selected, cfg)
            if not components:
                continue
            returns.append(components["gross_return"] - cfg.one_way_cost * float(turnover))
        rows.append(
            {
                "iteration": iteration,
                "null_type": "random_negative_funding_basket_observed_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def summarize_v134(
    portfolio: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V134Config = V134Config(),
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
        "median_coverage": float(portfolio["coverage"].median()),
        "median_eligible_negative_names": float(portfolio["eligible_negative_names"].median()),
        "mean_turnover": float(portfolio["realized_turnover"].mean()),
        "mean_long_notional": float(portfolio["long_notional"].mean()),
        "mean_btc_short_notional": float(portfolio["btc_short_notional"].mean()),
        "mean_price_bp": float(portfolio["price_return"].mean() * 10_000),
        "mean_funding_bp": float(portfolio["funding_return"].mean() * 10_000),
        "mean_gross_bp": float(portfolio["gross_return"].mean() * 10_000),
        "mean_primary_net_bp": observed * 10_000,
        "mean_stress_net_bp": float(portfolio["stress_net_return"].mean() * 10_000),
        "development_primary_net_bp": float(periods.get("development", np.nan) * 10_000),
        "validation_primary_net_bp": float(periods.get("validation", np.nan) * 10_000),
        "holdout_primary_net_bp": float(periods.get("holdout", np.nan) * 10_000),
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
                "bootstrap_95_low_bp",
            )
        )
        and row["null_percentile"] >= 90
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and row["max_abs_residual_btc_beta"] <= 1e-12
    )
    return pd.DataFrame([row])


def write_v134_negative_funding_beta_neutral_rebound(
    cfg: V134Config = V134Config(),
) -> dict[str, Path]:
    membership = load_v133_membership()
    prices = hourly_bybit_prices(load_v132_bybit_klines())
    member_funding = _combined_funding(
        RAW_BYBIT_ROOT / "funding", RECENT_ROOT / "bybit_funding", "symbol"
    )
    funding = pd.concat([member_funding, _load_btc_funding()], ignore_index=True)
    funding = funding.drop_duplicates(["symbol", "funding_time"], keep="last")
    panel = build_v134_weekly_panel(funding, prices, membership, cfg)
    portfolio = build_v134_portfolio(panel, cfg)
    nulls = build_v134_nulls(panel, portfolio, cfg)
    summary = summarize_v134(portfolio, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v134_negative_funding_beta_neutral_rebound_findings_2026_07_15.md"),
    }
    panel.to_parquet(paths["panel"], index=False)
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
                "# v13.4 Negative-Funding Beta-Neutral Rebound Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Signal, beta, prices, and funding are strictly causal. Gross notional",
                "is one and the estimated BTC beta is algebraically neutralized. No",
                "PaperLive, leverage, or status permission changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
