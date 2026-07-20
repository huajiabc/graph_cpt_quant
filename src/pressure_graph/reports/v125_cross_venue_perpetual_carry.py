"""Weekly same-coin Bybit/Binance perpetual funding-spread carry."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v121_top_trader_community_rotation import (
    _membership,
    _period,
)
from pressure_graph.reports.v123_cross_sectional_funding_carry import (
    load_v123_funding,
    load_v123_prices,
)


REPORT_ROOT = Path("reports/v12_5_cross_venue_perpetual_carry")
BINANCE_ROOT = Path("data/external/binance_um_carry")
BYBIT_FUNDING_ROOT = Path("data/raw/bybit/funding")
BYBIT_FEATURE_PATH = Path(
    "data/processed/v0_3/perp_pressure_features_all_eligible.parquet"
)
MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
CANDIDATES = (
    "CV1_7D_FUNDING_SPREAD",
    "CV2_30D_FUNDING_SPREAD",
    "CV3_COMMUNITY_FUNDING_SPREAD",
)


@dataclass(frozen=True)
class V125Config:
    binance_root: Path = BINANCE_ROOT
    funding_root: Path = BYBIT_FUNDING_ROOT
    feature_path: Path = BYBIT_FEATURE_PATH
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    first_entry: pd.Timestamp = pd.Timestamp("2025-08-04", tz="UTC")
    holding_days: int = 7
    bucket_size: int = 9
    community_count: int = 8
    focal_cost: float = 0.004
    stress_cost: float = 0.008
    one_way_cost: float = 0.002
    direct_null_iterations: int = 500
    community_null_iterations: int = 200
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def load_v125_binance_prices(cfg: V125Config = V125Config()) -> pd.DataFrame:
    frames = []
    for symbol in sorted(_membership(cfg)["symbol"].astype(str).unique()):
        path = cfg.binance_root / "klines_1h" / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path, columns=["bybit_symbol", "feature_time", "close"]
        ).rename(columns={"bybit_symbol": "symbol", "close": "binance_close"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True)
    prices["feature_time"] = pd.to_datetime(
        prices["feature_time"], utc=True, errors="coerce"
    )
    prices["binance_close"] = pd.to_numeric(
        prices["binance_close"], errors="coerce"
    )
    return (
        prices.dropna(subset=["symbol", "feature_time", "binance_close"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def load_v125_binance_funding(cfg: V125Config = V125Config()) -> pd.DataFrame:
    frames = []
    for symbol in sorted(_membership(cfg)["symbol"].astype(str).unique()):
        path = cfg.binance_root / "funding" / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path,
            columns=["bybit_symbol", "funding_time", "funding_rate_settled"],
        ).rename(columns={"bybit_symbol": "symbol"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    funding = pd.concat(frames, ignore_index=True)
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], utc=True, errors="coerce"
    ).dt.floor("s")
    funding["funding_rate_settled"] = pd.to_numeric(
        funding["funding_rate_settled"], errors="coerce"
    )
    return (
        funding.dropna(subset=["symbol", "funding_time", "funding_rate_settled"])
        .drop_duplicates(["symbol", "funding_time"], keep="last")
        .sort_values(["funding_time", "symbol"])
        .reset_index(drop=True)
    )


def build_v125_weekly_panel(
    bybit_funding: pd.DataFrame,
    binance_funding: pd.DataFrame,
    bybit_prices: pd.DataFrame,
    binance_prices: pd.DataFrame,
    cfg: V125Config = V125Config(),
) -> pd.DataFrame:
    membership = _membership(cfg)
    bybit_close = bybit_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    binance_close = binance_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="binance_close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    if bybit_close.empty or binance_close.empty:
        return pd.DataFrame()
    last_price_time = min(bybit_close.index.max(), binance_close.index.max())
    entries = pd.date_range(
        cfg.first_entry,
        last_price_time - pd.Timedelta(days=cfg.holding_days),
        freq="7D",
        tz="UTC",
    )
    bybit_groups = {
        symbol: group.sort_values("funding_time")
        for symbol, group in bybit_funding.groupby("symbol", observed=True)
    }
    binance_groups = {
        symbol: group.sort_values("funding_time")
        for symbol, group in binance_funding.groupby("symbol", observed=True)
    }
    rows = []
    for entry in entries:
        exit_time = entry + pd.Timedelta(days=cfg.holding_days)
        if (
            entry not in bybit_close.index
            or exit_time not in bybit_close.index
            or entry not in binance_close.index
            or exit_time not in binance_close.index
        ):
            continue
        month = entry.floor("D").replace(day=1)
        local_membership = membership[membership["month_start"].eq(month)]
        for item in local_membership.itertuples(index=False):
            symbol = str(item.symbol)
            if symbol not in bybit_close.columns or symbol not in binance_close.columns:
                continue
            prices = (
                bybit_close.loc[entry, symbol],
                bybit_close.loc[exit_time, symbol],
                binance_close.loc[entry, symbol],
                binance_close.loc[exit_time, symbol],
            )
            if not all(np.isfinite(value) and value > 0 for value in prices):
                continue
            bybit = bybit_groups.get(symbol)
            binance = binance_groups.get(symbol)
            if bybit is None or binance is None:
                continue
            bt, br = bybit["funding_time"], bybit["funding_rate_settled"]
            nt, nr = binance["funding_time"], binance["funding_rate_settled"]
            bybit_7d = br[bt.ge(entry - pd.Timedelta(days=7)) & bt.lt(entry)]
            binance_7d = nr[nt.ge(entry - pd.Timedelta(days=7)) & nt.lt(entry)]
            bybit_30d = br[bt.ge(entry - pd.Timedelta(days=30)) & bt.lt(entry)]
            binance_30d = nr[nt.ge(entry - pd.Timedelta(days=30)) & nt.lt(entry)]
            if any(
                window.empty
                for window in (bybit_7d, binance_7d, bybit_30d, binance_30d)
            ):
                continue
            future_bybit = float(br[bt.gt(entry) & bt.le(exit_time)].sum())
            future_binance = float(nr[nt.gt(entry) & nt.le(exit_time)].sum())
            bybit_return = float(prices[1] / prices[0] - 1.0)
            binance_return = float(prices[3] / prices[2] - 1.0)
            price_basis = bybit_return - binance_return
            funding_spread = future_binance - future_bybit
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "month_start": month,
                    "period": _period(entry),
                    "community_id": str(item.community_id),
                    "symbol": symbol,
                    "score_7d": float(binance_7d.sum() - bybit_7d.sum()),
                    "score_30d": float(binance_30d.sum() - bybit_30d.sum()),
                    "future_bybit_funding": future_bybit,
                    "future_binance_funding": future_binance,
                    "funding_spread_return": funding_spread,
                    "bybit_return": bybit_return,
                    "binance_return": binance_return,
                    "price_basis_return": price_basis,
                    "pair_gross_return": price_basis + funding_spread,
                }
            )
    return pd.DataFrame(rows)


def _top_positive_weights(
    local: pd.DataFrame, score_column: str, bucket_size: int
) -> tuple[dict[str, float], list[str]]:
    usable = local.dropna(subset=[score_column, "pair_gross_return"])
    usable = usable[usable[score_column].gt(0)].sort_values(
        [score_column, "symbol"]
    )
    selected = usable.tail(bucket_size)["symbol"].astype(str).tolist()
    if len(selected) < bucket_size:
        return {}, []
    return {symbol: 1.0 / bucket_size for symbol in selected}, selected


def _components(local: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    indexed = local.set_index("symbol")
    return {
        name: float(
            sum(weights[symbol] * indexed.loc[symbol, column] for symbol in weights)
        )
        for name, column in (
            ("bybit_return", "bybit_return"),
            ("binance_return", "binance_return"),
            ("price_basis_return", "price_basis_return"),
            ("funding_spread_return", "funding_spread_return"),
            ("gross_return", "pair_gross_return"),
        )
    }


def build_v125_portfolios(
    panel: pd.DataFrame, cfg: V125Config = V125Config()
) -> pd.DataFrame:
    rows = []
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        for candidate, score_column in zip(CANDIDATES[:2], ("score_7d", "score_30d")):
            weights, selected = _top_positive_weights(
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
                    "selected_symbols": "|".join(selected),
                    "_weights": weights,
                    **_components(local, weights),
                }
            )
        community_selected = []
        for community, group in local.groupby("community_id", observed=True):
            usable = group.dropna(subset=["score_7d", "pair_gross_return"])
            usable = usable[usable["score_7d"].gt(0)].sort_values(
                ["score_7d", "symbol"]
            )
            if usable.empty:
                continue
            community_selected.append((str(community), str(usable.iloc[-1]["symbol"])))
        if len(community_selected) != cfg.community_count:
            continue
        weights = {
            symbol: 1.0 / cfg.community_count for _, symbol in community_selected
        }
        if len(weights) != cfg.community_count:
            continue
        rows.append(
            {
                "candidate": CANDIDATES[2],
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "selected_symbols": "|".join(
                    symbol for _, symbol in community_selected
                ),
                "_weights": weights,
                **_components(local, weights),
            }
        )
    return pd.DataFrame(rows)


def apply_v125_costs(
    portfolios: pd.DataFrame, cfg: V125Config = V125Config()
) -> pd.DataFrame:
    output = portfolios.copy()
    output["net_40bp"] = output["gross_return"] - cfg.focal_cost
    output["net_80bp"] = output["gross_return"] - cfg.stress_cost
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


def build_v125_nulls(
    panel: pd.DataFrame, cfg: V125Config = V125Config()
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    weeks = [group.copy() for _, group in panel.groupby("entry_time", sort=True)]
    rows = []
    for candidate, score_column in zip(CANDIDATES[:2], ("score_7d", "score_30d")):
        usable_returns = []
        for local in weeks:
            usable = local.dropna(subset=[score_column, "pair_gross_return"])
            usable = usable[usable[score_column].gt(0)]
            if len(usable) >= cfg.bucket_size:
                usable_returns.append(usable["pair_gross_return"].to_numpy(dtype=float))
        for iteration in range(cfg.direct_null_iterations):
            returns = [
                float(
                    rng.choice(values, size=cfg.bucket_size, replace=False).mean()
                )
                for values in usable_returns
            ]
            rows.append(
                {
                    "candidate": candidate,
                    "iteration": iteration,
                    "null_type": "within_week_random_positive_spread",
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
            randomized = local.assign(
                random_community=local["symbol"].map(assignments[month])
            )
            selected = []
            for _, group in randomized.groupby("random_community", observed=True):
                usable = group.dropna(subset=["score_7d", "pair_gross_return"])
                usable = usable[usable["score_7d"].gt(0)].sort_values(
                    ["score_7d", "symbol"]
                )
                if usable.empty:
                    continue
                selected.append(float(usable.iloc[-1]["pair_gross_return"]))
            if len(selected) == cfg.community_count:
                returns.append(float(np.mean(selected)))
        rows.append(
            {
                "candidate": CANDIDATES[2],
                "iteration": iteration,
                "null_type": "random_monthly_communities_top_positive",
                "mean_net_40bp": float(np.mean(returns) - cfg.focal_cost),
            }
        )
    return pd.DataFrame(rows)


def summarize_v125(
    portfolios: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V125Config = V125Config(),
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
            "mean_bybit_bp": float(local["bybit_return"].mean() * 10_000),
            "mean_binance_bp": float(local["binance_return"].mean() * 10_000),
            "mean_price_basis_bp": float(
                local["price_basis_return"].mean() * 10_000
            ),
            "mean_funding_spread_bp": float(
                local["funding_spread_return"].mean() * 10_000
            ),
            "mean_gross_bp": float(local["gross_return"].mean() * 10_000),
            "mean_net_40bp_bp": observed * 10_000,
            "mean_net_80bp_bp": float(local["net_80bp"].mean() * 10_000),
            "mean_turnover_net_bp": float(
                local["turnover_net_20bp_oneway"].mean() * 10_000
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
            and row["mean_funding_spread_bp"] > 0
            and all(
                row[key] > 0
                for key in (
                    "development_net_40bp_bp",
                    "validation_net_40bp_bp",
                    "holdout_net_40bp_bp",
                    "mean_net_80bp_bp",
                    "bootstrap_95_low_bp",
                )
            )
            and row["null_percentile"] >= 90
            and row["positive_month_concentration"] <= 0.35
            and row["worst_period_bp"] >= -40
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_v125_cross_venue_perpetual_carry(
    cfg: V125Config = V125Config(),
) -> dict[str, Path]:
    bybit_funding = load_v123_funding(cfg)
    binance_funding = load_v125_binance_funding(cfg)
    bybit_prices = load_v123_prices(cfg)
    binance_prices = load_v125_binance_prices(cfg)
    panel = build_v125_weekly_panel(
        bybit_funding, binance_funding, bybit_prices, binance_prices, cfg
    )
    portfolios = apply_v125_costs(build_v125_portfolios(panel, cfg), cfg)
    nulls = build_v125_nulls(panel, cfg)
    summary = summarize_v125(portfolios, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolios": root / "weekly_portfolios.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v125_cross_venue_perpetual_carry_findings_2026_07_15.md"
        ),
    }
    panel.to_parquet(paths["panel"], index=False)
    portfolios.drop(columns="_weights").to_parquet(paths["portfolios"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "bybit_funding_rows": len(bybit_funding),
                "binance_funding_rows": len(binance_funding),
                "binance_symbols": int(binance_prices["symbol"].nunique()),
                "panel_rows": len(panel),
                "weeks": int(panel["entry_time"].nunique()),
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_candidate" if promoted else "reject_all_as_tradable_alpha"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v12.5 Cross-Venue Same-Coin Perpetual Carry Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Both venue prices use fully closed bars. Funding scores are strictly "
                "pre-entry and realized funding is `(entry, exit]`. No existing PaperLive "
                "strategy was changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
