"""Weekly cross-sectional perpetual funding carry portfolios."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import estimate_v106_betas
from pressure_graph.reports.v121_top_trader_community_rotation import (
    BTC,
    _membership,
    _period,
)


REPORT_ROOT = Path("reports/v12_3_cross_sectional_funding_carry")
FUNDING_ROOT = Path("data/raw/bybit/funding")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
CANDIDATES = (
    "FC1_7D_FUNDING_CARRY",
    "FC2_30D_FUNDING_CARRY",
    "FC3_COMMUNITY_NEUTRAL_CARRY",
)


@dataclass(frozen=True)
class V123Config:
    funding_root: Path = FUNDING_ROOT
    feature_path: Path = FEATURE_PATH
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    first_entry: pd.Timestamp = pd.Timestamp("2025-08-04", tz="UTC")
    holding_days: int = 7
    bucket_size: int = 9
    minimum_cross_section: int = 48
    minimum_community_coverage: int = 6
    focal_cost: float = 0.004
    stress_cost: float = 0.006
    one_way_cost: float = 0.002
    direct_null_iterations: int = 500
    community_null_iterations: int = 200
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def load_v123_funding(cfg: V123Config = V123Config()) -> pd.DataFrame:
    symbols = set(_membership(cfg)["symbol"])
    frames = []
    for symbol in sorted(symbols):
        path = cfg.funding_root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path,
            columns=["symbol", "funding_time", "funding_rate_settled"],
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    funding = pd.concat(frames, ignore_index=True)
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], utc=True, errors="coerce"
    )
    funding["funding_rate_settled"] = pd.to_numeric(
        funding["funding_rate_settled"], errors="coerce"
    )
    return (
        funding.dropna(subset=["symbol", "funding_time", "funding_rate_settled"])
        .drop_duplicates(["symbol", "funding_time"], keep="last")
        .sort_values(["funding_time", "symbol"])
        .reset_index(drop=True)
    )


def load_v123_prices(cfg: V123Config = V123Config()) -> pd.DataFrame:
    symbols = set(_membership(cfg)["symbol"]) | {BTC}
    columns = ["symbol", "feature_time", "close", "ret_1h", "warmup_complete"]
    parquet = pq.ParquetFile(cfg.feature_path)
    frames = []
    for index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(index, columns=columns).to_pandas()
        chunk["feature_time"] = pd.to_datetime(
            chunk["feature_time"], utc=True, errors="coerce"
        )
        chunk = chunk[
            chunk["symbol"].astype(str).isin(symbols)
            & chunk["feature_time"].dt.minute.eq(0)
            & chunk["warmup_complete"].fillna(False).astype(bool)
        ].copy()
        if not chunk.empty:
            frames.append(chunk.drop(columns="warmup_complete"))
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["ret_1h"] = pd.to_numeric(prices["ret_1h"], errors="coerce")
    return (
        prices.dropna(subset=["symbol", "feature_time", "close"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def _monthly_betas(
    prices: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_month in sorted(membership["month_start"].unique()):
        month = pd.Timestamp(raw_month)
        symbols = set(membership.loc[membership["month_start"].eq(month), "symbol"])
        history = prices[
            prices["feature_time"].ge(month - pd.Timedelta(days=30))
            & prices["feature_time"].lt(month)
            & prices["symbol"].isin(symbols | {BTC})
        ]
        pivot = history.pivot_table(
            index="feature_time",
            columns="symbol",
            values="ret_1h",
            aggfunc="last",
            observed=True,
        )
        betas = estimate_v106_betas(pivot)
        for symbol in sorted(symbols & set(betas.index.astype(str))):
            rows.append(
                {"month_start": month, "symbol": symbol, "btc_beta": betas[symbol]}
            )
    return pd.DataFrame(rows)


def build_v123_weekly_panel(
    funding: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V123Config = V123Config(),
) -> pd.DataFrame:
    membership = _membership(cfg)
    betas = _monthly_betas(prices, membership)
    close = prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    last_entry = close.index.max() - pd.Timedelta(days=cfg.holding_days)
    entries = pd.date_range(cfg.first_entry, last_entry, freq="7D", tz="UTC")
    rows = []
    funding_by_symbol = {
        symbol: group.sort_values("funding_time")
        for symbol, group in funding.groupby("symbol", observed=True)
    }
    beta_lookup = betas.set_index(["month_start", "symbol"])["btc_beta"]
    for entry in entries:
        exit_time = entry + pd.Timedelta(days=cfg.holding_days)
        if entry not in close.index or exit_time not in close.index or BTC not in close.columns:
            continue
        month = entry.floor("D").replace(day=1)
        local_membership = membership[membership["month_start"].eq(month)]
        if local_membership.empty:
            continue
        btc_return = float(close.loc[exit_time, BTC] / close.loc[entry, BTC] - 1.0)
        for item in local_membership.itertuples(index=False):
            symbol = str(item.symbol)
            if symbol not in close.columns:
                continue
            entry_price = close.loc[entry, symbol]
            exit_price = close.loc[exit_time, symbol]
            if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
                continue
            symbol_funding = funding_by_symbol.get(symbol)
            if symbol_funding is None:
                continue
            rate = symbol_funding["funding_rate_settled"]
            time = symbol_funding["funding_time"]
            score_7d = float(rate[time.ge(entry - pd.Timedelta(days=7)) & time.lt(entry)].sum())
            history_30d = rate[
                time.ge(entry - pd.Timedelta(days=30)) & time.lt(entry)
            ]
            if history_30d.empty:
                continue
            score_30d = float(history_30d.mean())
            future_funding = float(rate[time.gt(entry) & time.le(exit_time)].sum())
            price_return = float(exit_price / entry_price - 1.0)
            beta = beta_lookup.get((month, symbol), np.nan)
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "month_start": month,
                    "period": _period(entry),
                    "community_id": str(item.community_id),
                    "symbol": symbol,
                    "score_7d": score_7d,
                    "score_30d": score_30d,
                    "future_funding": future_funding,
                    "price_return": price_return,
                    "btc_beta": beta,
                    "residual_price_return": price_return - beta * btc_return,
                    "carry_adjusted_return": price_return - future_funding,
                    "residual_carry_adjusted_return": price_return
                    - beta * btc_return
                    - future_funding,
                }
            )
    return pd.DataFrame(rows)


def _direct_weights(
    local: pd.DataFrame, score_column: str, bucket_size: int
) -> tuple[dict[str, float], list[str], list[str]]:
    ranked = local.dropna(
        subset=[score_column, "carry_adjusted_return", "residual_carry_adjusted_return"]
    ).sort_values([score_column, "symbol"])
    low = ranked.head(bucket_size)["symbol"].astype(str).tolist()
    high = ranked.tail(bucket_size)["symbol"].astype(str).tolist()
    if len(low) < bucket_size or len(high) < bucket_size or set(low) & set(high):
        return {}, [], []
    weights = {symbol: 0.5 / len(low) for symbol in low}
    weights.update({symbol: -0.5 / len(high) for symbol in high})
    return weights, low, high


def _portfolio_components(
    local: pd.DataFrame, weights: dict[str, float]
) -> dict[str, float]:
    indexed = local.set_index("symbol")
    price = sum(weights[symbol] * indexed.loc[symbol, "price_return"] for symbol in weights)
    funding = sum(
        -weights[symbol] * indexed.loc[symbol, "future_funding"] for symbol in weights
    )
    residual_price = sum(
        weights[symbol] * indexed.loc[symbol, "residual_price_return"]
        for symbol in weights
    )
    return {
        "price_return": float(price),
        "funding_return": float(funding),
        "gross_return": float(price + funding),
        "residual_gross_return": float(residual_price + funding),
    }


def build_v123_portfolios(
    panel: pd.DataFrame,
    cfg: V123Config = V123Config(),
) -> pd.DataFrame:
    rows = []
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        if len(local) < cfg.minimum_cross_section:
            continue
        for candidate, score_column in (
            (CANDIDATES[0], "score_7d"),
            (CANDIDATES[1], "score_30d"),
        ):
            weights, long_symbols, short_symbols = _direct_weights(
                local, score_column, cfg.bucket_size
            )
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
                    "long_symbols": "|".join(long_symbols),
                    "short_symbols": "|".join(short_symbols),
                    "_weights": weights,
                    **_portfolio_components(local, weights),
                }
            )

        community_pairs = []
        for community, group in local.groupby("community_id", observed=True):
            usable = group.dropna(subset=["score_7d", "carry_adjusted_return"]).sort_values(
                ["score_7d", "symbol"]
            )
            if len(usable) < cfg.minimum_community_coverage:
                continue
            community_pairs.append(
                (community, str(usable.iloc[0]["symbol"]), str(usable.iloc[-1]["symbol"]))
            )
        if len(community_pairs) != 8:
            continue
        weights = {}
        for _, long_symbol, short_symbol in community_pairs:
            weights[long_symbol] = weights.get(long_symbol, 0.0) + 0.5 / len(community_pairs)
            weights[short_symbol] = weights.get(short_symbol, 0.0) - 0.5 / len(community_pairs)
        rows.append(
            {
                "candidate": CANDIDATES[2],
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "long_symbols": "|".join(pair[1] for pair in community_pairs),
                "short_symbols": "|".join(pair[2] for pair in community_pairs),
                "_weights": weights,
                **_portfolio_components(local, weights),
            }
        )
    return pd.DataFrame(rows)


def apply_v123_costs(
    portfolios: pd.DataFrame,
    cfg: V123Config = V123Config(),
) -> pd.DataFrame:
    output = portfolios.copy()
    output["net_40bp"] = output["gross_return"] - cfg.focal_cost
    output["net_60bp"] = output["gross_return"] - cfg.stress_cost
    output["residual_net_40bp"] = output["residual_gross_return"] - cfg.focal_cost
    output["realized_turnover"] = np.nan
    output["turnover_net_20bp_oneway"] = np.nan
    for _, indices in output.groupby("candidate", sort=True).groups.items():
        ordered = output.loc[indices].sort_values("entry_time")
        previous: dict[str, float] | None = None
        turnovers = []
        for index, row in ordered.iterrows():
            current = row["_weights"]
            if previous is None:
                turnover = 1.0
            else:
                symbols = set(previous) | set(current)
                turnover = sum(
                    abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
                    for symbol in symbols
                )
            turnovers.append((index, turnover))
            previous = current
        if turnovers:
            index, turnover = turnovers[-1]
            turnovers[-1] = (index, turnover + 1.0)
        for index, turnover in turnovers:
            output.loc[index, "realized_turnover"] = turnover
            output.loc[index, "turnover_net_20bp_oneway"] = (
                output.loc[index, "gross_return"] - cfg.one_way_cost * turnover
            )
    return output


def build_v123_nulls(
    panel: pd.DataFrame,
    cfg: V123Config = V123Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    weeks = [group.copy() for _, group in panel.groupby("entry_time", sort=True)]
    rows = []
    for candidate in CANDIDATES[:2]:
        for iteration in range(cfg.direct_null_iterations):
            returns = []
            for local in weeks:
                usable = local.dropna(subset=["carry_adjusted_return"])
                if len(usable) < 2 * cfg.bucket_size:
                    continue
                order = rng.permutation(len(usable))
                long = usable.iloc[order[: cfg.bucket_size]]["carry_adjusted_return"].mean()
                short = usable.iloc[order[cfg.bucket_size : 2 * cfg.bucket_size]][
                    "carry_adjusted_return"
                ].mean()
                returns.append(0.5 * (long - short))
            rows.append(
                {
                    "candidate": candidate,
                    "iteration": iteration,
                    "null_type": "within_week_random_bucket",
                    "mean_net_40bp": float(np.mean(returns) - cfg.focal_cost),
                }
            )

    months = {
        pd.Timestamp(month): sorted(group["symbol"].astype(str).unique())
        for month, group in _membership(cfg).groupby("month_start", observed=True)
    }
    for iteration in range(cfg.community_null_iterations):
        assignments = {}
        for month, symbols in months.items():
            shuffled = np.asarray(symbols)[rng.permutation(len(symbols))]
            assignments[month] = {
                str(symbol): int(index // 9) for index, symbol in enumerate(shuffled)
            }
        returns = []
        for local in weeks:
            month = pd.Timestamp(local["month_start"].iloc[0])
            local = local.assign(
                random_community=local["symbol"].map(assignments[month])
            )
            weights = {}
            for _, group in local.groupby("random_community", observed=True):
                usable = group.dropna(subset=["score_7d", "carry_adjusted_return"]).sort_values(
                    ["score_7d", "symbol"]
                )
                if len(usable) < cfg.minimum_community_coverage:
                    continue
                long_symbol = str(usable.iloc[0]["symbol"])
                short_symbol = str(usable.iloc[-1]["symbol"])
                weights[long_symbol] = weights.get(long_symbol, 0.0) + 0.5 / 8
                weights[short_symbol] = weights.get(short_symbol, 0.0) - 0.5 / 8
            if len(weights) < 12:
                continue
            indexed = local.set_index("symbol")["carry_adjusted_return"]
            returns.append(sum(weights[symbol] * indexed[symbol] for symbol in weights))
        rows.append(
            {
                "candidate": CANDIDATES[2],
                "iteration": iteration,
                "null_type": "random_monthly_communities",
                "mean_net_40bp": float(np.mean(returns) - cfg.focal_cost),
            }
        )
    return pd.DataFrame(rows)


def summarize_v123(
    portfolios: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V123Config = V123Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 2)
    rows = []
    for candidate, local in portfolios.groupby("candidate", sort=True):
        values = local["net_40bp"].to_numpy(dtype=float)
        draws = rng.choice(
            values, size=(cfg.bootstrap_iterations, len(values)), replace=True
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        periods = local.groupby("period", observed=True)["net_40bp"].mean()
        months = local.groupby("month_start", observed=True)["net_40bp"].sum()
        positive = months[months.gt(0)]
        concentration = (
            float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
        )
        candidate_null = nulls.loc[
            nulls["candidate"].eq(candidate), "mean_net_40bp"
        ]
        observed = float(local["net_40bp"].mean())
        counts = local["period"].value_counts()
        row = {
            "candidate": candidate,
            "weeks": len(local),
            "months": int(local["month_start"].nunique()),
            "validation_weeks": int(counts.get("validation", 0)),
            "holdout_weeks": int(counts.get("holdout", 0)),
            "median_coverage": float(local["coverage"].median()),
            "mean_price_bp": float(local["price_return"].mean() * 10_000),
            "mean_funding_bp": float(local["funding_return"].mean() * 10_000),
            "mean_gross_bp": float(local["gross_return"].mean() * 10_000),
            "mean_net_40bp_bp": observed * 10_000,
            "mean_net_60bp_bp": float(local["net_60bp"].mean() * 10_000),
            "mean_turnover_net_bp": float(
                local["turnover_net_20bp_oneway"].mean() * 10_000
            ),
            "mean_residual_net_40bp_bp": float(
                local["residual_net_40bp"].mean() * 10_000
            ),
            "development_net_40bp_bp": float(
                periods.get("development", np.nan) * 10_000
            ),
            "validation_net_40bp_bp": float(
                periods.get("validation", np.nan) * 10_000
            ),
            "holdout_net_40bp_bp": float(periods.get("holdout", np.nan) * 10_000),
            "bootstrap_95_low_bp": float(ci_low * 10_000),
            "bootstrap_95_high_bp": float(ci_high * 10_000),
            "null_percentile": float(100 * candidate_null.le(observed).mean()),
            "positive_month_concentration": concentration,
            "worst_period_bp": float(periods.min() * 10_000),
        }
        row["promote"] = bool(
            row["weeks"] >= 40
            and row["months"] >= 10
            and row["validation_weeks"] >= 10
            and row["holdout_weeks"] >= 8
            and row["mean_funding_bp"] > 0
            and all(
                row[key] > 0
                for key in (
                    "development_net_40bp_bp",
                    "validation_net_40bp_bp",
                    "holdout_net_40bp_bp",
                    "mean_net_60bp_bp",
                    "mean_residual_net_40bp_bp",
                    "bootstrap_95_low_bp",
                )
            )
            and row["null_percentile"] >= 90
            and row["positive_month_concentration"] <= 0.35
            and row["worst_period_bp"] >= -40
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_v123_cross_sectional_funding_carry(
    cfg: V123Config = V123Config(),
) -> dict[str, Path]:
    funding = load_v123_funding(cfg)
    prices = load_v123_prices(cfg)
    panel = build_v123_weekly_panel(funding, prices, cfg)
    portfolios = apply_v123_costs(build_v123_portfolios(panel, cfg), cfg)
    nulls = build_v123_nulls(panel, cfg)
    summary = summarize_v123(portfolios, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolios": root / "weekly_portfolios.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v123_cross_sectional_funding_carry_findings_2026_07_15.md"),
    }
    panel.to_parquet(paths["panel"], index=False)
    portfolios.drop(columns="_weights").to_parquet(paths["portfolios"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "funding_rows": len(funding),
                "funding_symbols": int(funding["symbol"].nunique()),
                "panel_rows": len(panel),
                "weeks": int(panel["entry_time"].nunique()),
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_candidate" if promoted else "reject_all_as_tradable_alpha"
    findings = "\n".join(
        [
            "# v12.3 Cross-Sectional Funding Carry Findings",
            "",
            f"Verdict: `{verdict}`.",
            "",
            summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "Funding, price, and BTC-residual components use only as-of settlements and exact "
            "seven-day closes. No existing PaperLive strategy was changed.",
            "",
        ]
    )
    paths["findings"].write_text(findings, encoding="utf-8")
    return paths
